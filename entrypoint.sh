#!/bin/bash
set -e

echo "Fetching SSL certificates from AWS Secrets Manager..."

# Create certs directory
mkdir -p /etc/nginx/certs

# Fetch certificate
aws secretsmanager get-secret-value \
    --secret-id meraki-assistant/ssl \
    --region ap-southeast-2 \
    --query 'SecretString' \
    --output text | python3 -c "
import sys, json
secret = json.load(sys.stdin)
with open('/etc/nginx/certs/origin.pem', 'w') as f:
    f.write(secret['certificate'])
with open('/etc/nginx/certs/origin.key', 'w') as f:
    f.write(secret['private_key'])
print('Certificates written successfully')
"

echo "Starting Nginx..."
exec nginx -g 'daemon off;'