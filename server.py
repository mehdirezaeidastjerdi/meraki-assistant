"""
Meraki AI Assistant — FastAPI Backend with JWT Auth
Secrets loaded from .env file for local development.

Setup:
1. Copy .env.example to .env and fill in your values
2. To hash a password for .env, run:
   python3 -c "from passlib.context import CryptContext; print(CryptContext(schemes=['bcrypt']).hash('yourpassword'))"
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import requests
import anthropic
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("meraki-assistant")

# ---------------------------------------------------------------------------
# Config from .env
# ---------------------------------------------------------------------------
JWT_SECRET       = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_MINS  = int(os.getenv("JWT_EXPIRE_MINS", "480"))  # 8 hours

ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
MERAKI_KEY       = os.getenv("MERAKI_API_KEY", "")
MERAKI_URL       = os.getenv("MERAKI_BASE_URL", "https://api.meraki.com/api/v1")
CLAUDE_MODEL     = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Users stored as JSON in .env: USERS_JSON={"admin":"$2b$12$hashed..."}
USERS_DB: dict = json.loads(os.getenv("USERS_JSON", "{}"))

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def authenticate_user(username: str, password: str) -> Optional[str]:
    hashed = USERS_DB.get(username)
    if not hashed or not verify_password(password, hashed):
        return None
    return username

def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINS)
    return jwt.encode({"sub": username, "exp": expire}, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired session. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if not username or username not in USERS_DB:
            raise exc
        return username
    except JWTError:
        raise exc

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

def meraki_get(path: str) -> dict:
    r = requests.get(
        f"{MERAKI_URL}{path}",
        headers={"X-Cisco-Meraki-API-Key": MERAKI_KEY, "Content-Type": "application/json"},
        timeout=15
    )
    r.raise_for_status()
    return r.json()

def run_tool(name: str, inputs: dict) -> dict:
    try:
        match name:
            case "get_organizations":   return meraki_get("/organizations")
            case "get_networks":        return meraki_get(f"/organizations/{inputs['org_id']}/networks")
            case "get_devices":         return meraki_get(f"/networks/{inputs['network_id']}/devices")
            case "get_clients":         return meraki_get(f"/networks/{inputs['network_id']}/clients?timespan=3600")
            case "get_device_statuses": return meraki_get(f"/organizations/{inputs['org_id']}/devices/statuses")
            case "get_network_events":  return meraki_get(f"/networks/{inputs['network_id']}/events?perPage=50")
            case "get_uplink_statuses": return meraki_get(f"/organizations/{inputs['org_id']}/appliance/uplink/statuses")
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str

@app.post("/auth/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    username = authenticate_user(form.username, form.password)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_token(username)
    log.info(f"User '{username}' logged in")
    return {"access_token": token, "token_type": "bearer", "username": username}

@app.get("/auth/me")
async def me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}

# ---------------------------------------------------------------------------
# Chat endpoint — JWT protected
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")
async def chat(req: ChatRequest, current_user: str = Depends(get_current_user)):
    if not ANTHROPIC_KEY or not MERAKI_KEY:
        raise HTTPException(status_code=500, detail="Server API keys not configured. Check your .env file.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    tool_calls_log = []

    while True:
        response = client.messages.create(
            model=CLAUDE_MODEL,
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
                    result = run_tool(block.name, block.input)
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
async def root():
    return FileResponse("static/login.html")

@app.get("/chat-ui")
async def chat_ui():
    return FileResponse("static/index.html")
