#!/bin/bash

# Simple Token Update Script for Railway
# Run this locally every morning

echo "============================================================"
echo "🔄 UPSTOX TOKEN UPDATE FOR RAILWAY"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${RED}❌ Railway CLI not found!${NC}"
    echo -e "${YELLOW}Install it: npm i -g @railway/cli${NC}"
    exit 1
fi

echo -e "${YELLOW}📝 Get your new access token from Upstox:${NC}"
echo ""
echo "1. Visit: https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id=YOUR_API_KEY&redirect_uri=YOUR_REDIRECT_URI"
echo "2. Login and get the 'code' from redirect URL"
echo "3. Use this curl command to get token:"
echo ""
echo "curl -X POST 'https://api.upstox.com/v2/login/authorization/token' \\"
echo "  -H 'Content-Type: application/x-www-form-urlencoded' \\"
echo "  -d 'code=YOUR_CODE&client_id=YOUR_API_KEY&client_secret=YOUR_API_SECRET&redirect_uri=YOUR_REDIRECT_URI&grant_type=authorization_code'"
echo ""
echo "============================================================"
echo ""

# Get token from user
echo -e "${GREEN}Enter your new ACCESS_TOKEN:${NC}"
read -r NEW_TOKEN

if [ -z "$NEW_TOKEN" ]; then
    echo -e "${RED}❌ Token cannot be empty!${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}🔄 Updating Railway environment variable...${NC}"

# Update Railway
railway variables set UPSTOX_ACCESS_TOKEN="$NEW_TOKEN"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Token updated successfully on Railway!${NC}"
    echo ""
    echo -e "${YELLOW}📝 Next steps:${NC}"
    echo "1. Railway will auto-restart your bot"
    echo "2. Test with /start command on Telegram"
    echo "3. Set reminder for tomorrow morning!"
    echo ""
    echo -e "${GREEN}🎉 Done!${NC}"
else
    echo ""
    echo -e "${RED}❌ Failed to update Railway token${NC}"
    echo "Try manually: railway variables set UPSTOX_ACCESS_TOKEN=your_token"
fi
