"""
Daily Token Refresh Helper for Upstox API
Run this script every morning to get a new access token
"""

import os
import requests
from dotenv import load_dotenv, set_key

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_colored(text, color):
    print(f"{color}{text}{RESET}")

def generate_auth_url(api_key, redirect_uri):
    """Generate authorization URL"""
    base_url = "https://api.upstox.com/v2/login/authorization/dialog"
    params = f"?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}"
    return base_url + params

def get_access_token(auth_code, api_key, api_secret, redirect_uri):
    """Exchange authorization code for access token"""
    url = "https://api.upstox.com/v2/login/authorization/token"
    
    data = {
        'code': auth_code,
        'client_id': api_key,
        'client_secret': api_secret,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code'
    }
    
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print_colored(f"❌ Error: {e}", RED)
        return None

def update_env_file(token):
    """Update .env file with new token"""
    try:
        set_key('.env', 'UPSTOX_ACCESS_TOKEN', token)
        print_colored("✅ .env file updated successfully!", GREEN)
        return True
    except Exception as e:
        print_colored(f"❌ Error updating .env: {e}", RED)
        return False

def update_railway_token(token):
    """Update Railway environment variable"""
    try:
        result = os.system(f'railway variables set UPSTOX_ACCESS_TOKEN={token}')
        if result == 0:
            print_colored("✅ Railway token updated successfully!", GREEN)
            return True
        else:
            print_colored("⚠️  Railway CLI not found or error updating", YELLOW)
            return False
    except Exception as e:
        print_colored(f"❌ Error updating Railway: {e}", RED)
        return False

def main():
    print_colored("=" * 60, BLUE)
    print_colored("🔄 UPSTOX TOKEN REFRESH HELPER", BLUE)
    print_colored("=" * 60, BLUE)
    print()
    
    # Load environment variables
    load_dotenv()
    
    API_KEY = os.getenv('UPSTOX_API_KEY')
    API_SECRET = os.getenv('UPSTOX_API_SECRET')
    
    if not API_KEY or not API_SECRET:
        print_colored("❌ API_KEY or API_SECRET not found in .env file!", RED)
        return
    
    # Get redirect URI
    print_colored("📝 Enter your Redirect URI (from Upstox app settings):", YELLOW)
    print_colored("   Example: https://127.0.0.1 or https://yourwebsite.com/callback", YELLOW)
    redirect_uri = input("Redirect URI: ").strip()
    
    if not redirect_uri:
        print_colored("❌ Redirect URI is required!", RED)
        return
    
    # Generate authorization URL
    auth_url = generate_auth_url(API_KEY, redirect_uri)
    
    print()
    print_colored("=" * 60, BLUE)
    print_colored("STEP 1: Get Authorization Code", BLUE)
    print_colored("=" * 60, BLUE)
    print()
    print_colored("1. Open this URL in your browser:", GREEN)
    print_colored(auth_url, YELLOW)
    print()
    print_colored("2. Login with your Upstox credentials", GREEN)
    print_colored("3. After login, you'll be redirected to:", GREEN)
    print_colored(f"   {redirect_uri}?code=XXXXXX", YELLOW)
    print()
    print_colored("4. Copy the 'code' parameter from the URL", GREEN)
    print()
    
    # Get authorization code from user
    auth_code = input("Enter the authorization code: ").strip()
    
    if not auth_code:
        print_colored("❌ Authorization code is required!", RED)
        return
    
    print()
    print_colored("=" * 60, BLUE)
    print_colored("STEP 2: Exchange Code for Access Token", BLUE)
    print_colored("=" * 60, BLUE)
    print()
    
    # Get access token
    print_colored("🔄 Fetching access token...", YELLOW)
    token_data = get_access_token(auth_code, API_KEY, API_SECRET, redirect_uri)
    
    if not token_data or 'access_token' not in token_data:
        print_colored("❌ Failed to get access token!", RED)
        if token_data:
            print_colored(f"Response: {token_data}", RED)
        return
    
    access_token = token_data['access_token']
    
    print()
    print_colored("✅ Access Token Generated Successfully!", GREEN)
    print_colored("=" * 60, GREEN)
    print_colored(f"Token: {access_token[:20]}...{access_token[-20:]}", YELLOW)
    print_colored("=" * 60, GREEN)
    print()
    
    # Update .env file
    print_colored("=" * 60, BLUE)
    print_colored("STEP 3: Update Environment Variables", BLUE)
    print_colored("=" * 60, BLUE)
    print()
    
    update_env_file(access_token)
    
    # Ask if user wants to update Railway
    print()
    print_colored("Do you want to update Railway token as well? (y/n): ", YELLOW, )
    update_railway = input().strip().lower()
    
    if update_railway == 'y':
        update_railway_token(access_token)
    
    print()
    print_colored("=" * 60, GREEN)
    print_colored("✅ TOKEN REFRESH COMPLETED!", GREEN)
    print_colored("=" * 60, GREEN)
    print()
    print_colored("📝 Next Steps:", BLUE)
    print_colored("1. Restart your bot: python main.py", YELLOW)
    print_colored("2. Or restart Railway deployment", YELLOW)
    print_colored("3. Token will expire tomorrow at 3:30 AM", YELLOW)
    print_colored("4. Run this script again tomorrow morning", YELLOW)
    print()
    print_colored("⏰ Reminder: Set a daily alarm for token refresh!", YELLOW)
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_colored("\n👋 Token refresh cancelled by user", YELLOW)
    except Exception as e:
        print()
        print_colored(f"\n❌ Unexpected error: {e}", RED)
