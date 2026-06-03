"""
Meraki AI Assistant — FastAPI Backend
- AWS Cognito Hosted UI for authentication (Google, Microsoft, Email+Password)
- JWT token verification using Cognito public keys (JWKS)
- Per-user API keys stored in AWS SSM Parameter Store
- HTTP-only session cookie (expires on browser close)

Required .env values:
    COGNITO_USER_POOL_ID
    COGNITO_CLIENT_ID
    COGNITO_CLIENT_SECRET
    COGNITO_DOMAIN
    COGNITO_REGION
    APP_URL
    AWS_REGION
    SSM_PREFIX
"""

import json
import logging
import os
from functools import lru_cache
from typing import List, Optional

import boto3
import httpx
import requests
import anthropic
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("meraki-assistant")

# ---------------------------------------------------------------------------
# Cognito config
# ---------------------------------------------------------------------------
COGNITO_REGION        = os.getenv("COGNITO_REGION", "ap-southeast-2")
COGNITO_USER_POOL_ID  = os.getenv("COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID     = os.getenv("COGNITO_CLIENT_ID", "")
COGNITO_CLIENT_SECRET = os.getenv("COGNITO_CLIENT_SECRET", "")
COGNITO_DOMAIN        = os.getenv("COGNITO_DOMAIN", "")  # e.g. meraki-assistant.auth.ap-southeast-2.amazoncognito.com
APP_URL               = os.getenv("APP_URL", "http://localhost:8000")

COGNITO_ISSUER        = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
COGNITO_JWKS_URL      = f"{COGNITO_ISSUER}/.well-known/jwks.json"
COGNITO_TOKEN_URL     = f"https://{COGNITO_DOMAIN}/oauth2/token"
COGNITO_LOGOUT_URL    = f"https://{COGNITO_DOMAIN}/logout"
COGNITO_AUTH_URL      = f"https://{COGNITO_DOMAIN}/oauth2/authorize"
REDIRECT_URI          = f"{APP_URL}/auth/callback"

# ---------------------------------------------------------------------------
# SSM config
# ---------------------------------------------------------------------------
AWS_REGION  = os.getenv("AWS_REGION", "ap-southeast-2")
SSM_PREFIX  = os.getenv("SSM_PREFIX", "/meraki-assistant/users")

# ---------------------------------------------------------------------------
# JWKS cache — fetch Cognito public keys once
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_jwks():
    resp = httpx.get(COGNITO_JWKS_URL)
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------
def verify_cognito_token(token: str) -> dict:
    """Verify Cognito JWT token and return claims."""
    try:
        jwks = get_jwks()
        header = jwt.get_unverified_header(token)
        key = next((k for k in jwks["keys"] if k["kid"] == header["kid"]), None)
        if not key:
            raise HTTPException(status_code=401, detail="Invalid token key")
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=COGNITO_CLIENT_ID,
            issuer=COGNITO_ISSUER,
            options={"verify_at_hash": False}
        )
        return claims
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Token error: {str(e)}")

def get_current_user(session_token: Optional[str] = Cookie(None)) -> dict:
    """Extract and verify user from HTTP-only session cookie."""
    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return verify_cognito_token(session_token)

# ---------------------------------------------------------------------------
# SSM helpers
# ---------------------------------------------------------------------------
def ssm_client():
    return boto3.client("ssm", region_name=AWS_REGION)

def ssm_path(user_id: str, key: str) -> str:
    return f"{SSM_PREFIX}/{user_id}/{key}"

def ssm_get(user_id: str, key: str) -> Optional[str]:
    try:
        r = ssm_client().get_parameter(Name=ssm_path(user_id, key), WithDecryption=True)
        return r["Parameter"]["Value"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ParameterNotFound":
            return None
        raise

def ssm_put(user_id: str, key: str, value: str):
    ssm_client().put_parameter(
        Name=ssm_path(user_id, key),
        Value=value,
        Type="SecureString",
        Overwrite=True,
        Description=f"Meraki Assistant config for user: {user_id}"
    )

def load_user_config(user_id: str) -> dict:
    keys = ["anthropic_key", "meraki_key", "meraki_url", "model"]
    return {k: v for k in keys if (v := ssm_get(user_id, k))}

def save_user_config(user_id: str, config: dict):
    for k, v in config.items():
        if k in ["anthropic_key", "meraki_key", "meraki_url", "model"] and v:
            ssm_put(user_id, k, v)

# ---------------------------------------------------------------------------
# Meraki tools
# ---------------------------------------------------------------------------
TOOLS = [
    {"name": "get_organizations",   "description": "Get all Meraki organizations.",                                          "input_schema": {"type": "object", "properties": {},                                    "required": []}},
    {"name": "get_networks",        "description": "Get all networks in an organization.",                                   "input_schema": {"type": "object", "properties": {"org_id":     {"type": "string"}}, "required": ["org_id"]}},
    {"name": "get_devices",         "description": "Get all devices in a network.",                                         "input_schema": {"type": "object", "properties": {"network_id": {"type": "string"}}, "required": ["network_id"]}},
    {"name": "get_clients",         "description": "Get clients connected in the last hour.",                                "input_schema": {"type": "object", "properties": {"network_id": {"type": "string"}}, "required": ["network_id"]}},
    {"name": "get_device_statuses", "description": "Get online/offline status of all devices in an org.",                   "input_schema": {"type": "object", "properties": {"org_id":     {"type": "string"}}, "required": ["org_id"]}},
    {"name": "get_network_events",  "description": "Get recent network events: disconnections, DHCP failures, auth issues.", "input_schema": {"type": "object", "properties": {"network_id": {"type": "string"}}, "required": ["network_id"]}},
    {"name": "get_uplink_statuses", "description": "Get WAN uplink health for MX appliances.",                              "input_schema": {"type": "object", "properties": {"org_id":     {"type": "string"}}, "required": ["org_id"]}},
]

def meraki_get(path: str, meraki_key: str, meraki_url: str) -> dict:
    r = requests.get(
        f"{meraki_url}{path}",
        headers={"X-Cisco-Meraki-API-Key": meraki_key, "Content-Type": "application/json"},
        timeout=15
    )
    r.raise_for_status()
    return r.json()

def run_tool(name: str, inputs: dict, meraki_key: str, meraki_url: str) -> dict:
    try:
        match name:
            case "get_organizations":   return meraki_get("/organizations", meraki_key, meraki_url)
            case "get_networks":        return meraki_get(f"/organizations/{inputs['org_id']}/networks", meraki_key, meraki_url)
            case "get_devices":         return meraki_get(f"/networks/{inputs['network_id']}/devices", meraki_key, meraki_url)
            case "get_clients":         return meraki_get(f"/networks/{inputs['network_id']}/clients?timespan=3600", meraki_key, meraki_url)
            case "get_device_statuses": return meraki_get(f"/organizations/{inputs['org_id']}/devices/statuses", meraki_key, meraki_url)
            case "get_network_events":  return meraki_get(f"/networks/{inputs['network_id']}/events?perPage=50", meraki_key, meraki_url)
            case "get_uplink_statuses": return meraki_get(f"/organizations/{inputs['org_id']}/appliance/uplink/statuses", meraki_key, meraki_url)
            case _:                     return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        log.error(f"Tool {name} failed: {e}")
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Meraki AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[APP_URL],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.get("/auth/login")
async def login(provider: Optional[str] = None):
    """
    Without provider -> serve custom branded login page.
    With provider -> redirect straight to Cognito with that provider.
    e.g. /auth/login?provider=Google
    """
    if provider:
        url = (
            f"{COGNITO_AUTH_URL}"
            f"?client_id={COGNITO_CLIENT_ID}"
            f"&response_type=code"
            f"&scope=openid%20email%20profile"
            f"&redirect_uri={REDIRECT_URI}"
            f"&identity_provider={provider}"
            f"&prompt=select_account"
        )
        return RedirectResponse(url)
    return FileResponse("static/login.html")

@app.get("/auth/login/email")
async def login_email():
    """Redirect to Cognito for email/password login."""
    url = (
        f"{COGNITO_AUTH_URL}"
        f"?client_id={COGNITO_CLIENT_ID}"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
        f"&redirect_uri={REDIRECT_URI}"
    )
    return RedirectResponse(url)

@app.get("/auth/callback")
async def auth_callback(code: str):
    """Handle Cognito callback — exchange code for tokens and set cookie."""
    # Exchange authorization code for tokens
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            COGNITO_TOKEN_URL,
            data={
                "grant_type":   "authorization_code",
                "client_id":    COGNITO_CLIENT_ID,
                "client_secret": COGNITO_CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "code":         code,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if resp.status_code != 200:
        log.error(f"Token exchange failed: {resp.text}")
        raise HTTPException(status_code=400, detail="Authentication failed. Please try again.")

    tokens = resp.json()
    id_token = tokens.get("id_token")

    if not id_token:
        raise HTTPException(status_code=400, detail="No ID token received from Cognito.")

    # Verify token is valid
    claims = verify_cognito_token(id_token)
    log.info(f"User logged in: {claims.get('email')}")

    # Set HTTP-only cookie (no max_age = expires on browser close)
    response = RedirectResponse(url="/chat-ui", status_code=302)
    response.set_cookie(
        key="session_token",
        value=id_token,
        httponly=True,       # JS cannot read this cookie
        secure=False,        # Set to True in production (HTTPS)
        samesite="lax",
        max_age=3600,        # Expires when browser closes
    )
    log.info(f"Setting cookie for user: {claims.get('email')}")
    return response

@app.get("/auth/logout")
async def logout():
    """Clear session cookie and redirect to login page."""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("session_token")
    return response

@app.get("/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return current user info."""
    return {
        "username": current_user.get("email") or current_user.get("cognito:username"),
        "email": current_user.get("email"),
        "name": current_user.get("name"),
        "user_id": current_user.get("sub"),  # Cognito unique user ID
    }

# ---------------------------------------------------------------------------
# Config endpoints — load/save per-user keys from SSM
# ---------------------------------------------------------------------------
class UserConfig(BaseModel):
    anthropic_key: Optional[str] = None
    meraki_key: Optional[str] = None
    meraki_url: Optional[str] = None
    model: Optional[str] = None

@app.get("/config")
async def get_config(current_user: dict = Depends(get_current_user)):
    """Load this user's API keys from SSM. Returns masked values."""
    user_id = current_user.get("sub")
    try:
        cfg = load_user_config(user_id)
        masked = {}
        for k, v in cfg.items():
            if k in ("anthropic_key", "meraki_key") and v:
                masked[k] = "••••••••" + v[-4:]
            else:
                masked[k] = v
        return {
            "config": masked,
            "configured": bool(cfg.get("anthropic_key") and cfg.get("meraki_key"))
        }
    except Exception as e:
        log.error(f"Failed to load config for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not load config from AWS SSM.")

@app.post("/config")
async def save_config(cfg: UserConfig, current_user: dict = Depends(get_current_user)):
    """Save this user's API keys to SSM."""
    user_id = current_user.get("sub")
    try:
        save_user_config(user_id, cfg.dict(exclude_none=True))
        log.info(f"Config saved to SSM for user '{current_user.get('email')}'")
        return {"message": "Configuration saved successfully."}
    except Exception as e:
        log.error(f"Failed to save config for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not save config to AWS SSM.")

# ---------------------------------------------------------------------------
# Chat endpoint — cookie protected, keys from SSM
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user.get("sub")
    cfg = load_user_config(user_id)

    anthropic_key = cfg.get("anthropic_key")
    meraki_key    = cfg.get("meraki_key")
    meraki_url    = cfg.get("meraki_url") or "https://api.meraki.com/api/v1"
    model         = cfg.get("model") or "claude-haiku-4-5-20251001"

    if not anthropic_key or not meraki_key:
        raise HTTPException(status_code=400, detail="API keys not configured. Open Settings to add them.")

    client = anthropic.Anthropic(api_key=anthropic_key)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    tool_calls_log = []

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=(
                "You are a Meraki network assistant. Use tools to answer questions about the user's network. "
                "Be concise, clear, and highlight anything unusual or worth attention."
            ),
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls_log.append(block.name)
                    result = run_tool(block.name, block.input, meraki_key, meraki_url)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            reply = next((b.text for b in response.content if hasattr(b, "text")), "")
            return {"reply": reply, "tools_used": tool_calls_log}

# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root(session_token: Optional[str] = Cookie(None)):
    if session_token:
        try:
            verify_cognito_token(session_token)
            return RedirectResponse(url="/chat-ui", status_code=302)
        except:
            pass
    return FileResponse("static/login.html")

@app.get("/chat-ui")
async def chat_ui(session_token: Optional[str] = Cookie(None)):
    log.info(f"chat-ui hit, token present: {session_token is not None}")
    if not session_token:
        return RedirectResponse(url="/", status_code=302)
    try:
        claims = verify_cognito_token(session_token)
        log.info(f"chat-ui valid for: {claims.get('email')}")
        return FileResponse("static/index.html")
    except Exception as e:
        log.info(f"chat-ui token error: {e}")
        return RedirectResponse(url="/", status_code=302)
