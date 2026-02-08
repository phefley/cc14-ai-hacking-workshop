#!/usr/bin/env python3
"""
Script to update .env file with generated secrets and static configuration values.
"""
import os
import secrets
from pathlib import Path


def generate_secret_key():
    """Generate a random secret key using secrets module."""
    return secrets.token_hex(16)


def update_env_file():
    """Update .env file with generated secrets and static values."""
    env_file = Path(".env")
    
    # Check if .env file exists
    if not env_file.exists():
        print(f"Error: .env file not found in {Path.cwd()}")
        print(f"first, copy the .env.example file to .env, then run this script.")
        return False
    
    # Generate two secret keys
    secret_key = generate_secret_key()
    auth_key = generate_secret_key()
    
    # Define replacements
    replacements = {
        "YOUR_SECRET_KEY": secret_key,
        "YOUR_AUTH_KEY": auth_key,
        "https://YOUR_AOAI_ENDPOINT": "YOUR_AOAI_ENDPOINT",
        "YOUR_AOAI_API_KEY": "YOUR_AOAI_API_KEY"  # This keeps the placeholder as-is
    }
    
    # Read the .env file
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Perform replacements
    updated_content = content
    for old_value, new_value in replacements.items():
        if old_value in updated_content:
            updated_content = updated_content.replace(old_value, new_value)
            print(f"✓ Replaced '{old_value}'")
        else:
            print(f"⚠ Warning: '{old_value}' not found in .env file")
    
    # Write the updated content back to .env
    with open(env_file, 'w') as f:
        f.write(updated_content)
    
    print(f"\n✓ .env file updated successfully!")
    print(f"\nAccess the labs at http://localhost:5000/login?auth={auth_key}")
    return True


if __name__ == "__main__":
    update_env_file()
