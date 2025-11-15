#!/usr/bin/env python3
"""
Test Gemini API connectivity
"""

import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

def list_available_models():
    """List available Gemini models"""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        return []
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            models = result.get('models', [])
            print("📋 Available Gemini models:")
            available_models = []
            for model in models:
                model_name = model.get('name', '')
                display_name = model.get('displayName', '')
                supported_methods = model.get('supportedGenerationMethods', [])
                print(f"  - {model_name} ({display_name}) - Methods: {supported_methods}")
                if 'generateContent' in supported_methods:
                    available_models.append(model_name)
            
            return available_models
        else:
            print(f"❌ Error listing models: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

def test_gemini_api():
    """Test Gemini API connectivity"""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        return False
    
    # First list available models
    available_models = list_available_models()
    
    if not available_models:
        print("❌ No available models found")
        return False
    
    # Try the first available model that supports generateContent
    model_name = available_models[0] if available_models else "models/gemini-pro"
    print(f"\n🧪 Testing model: {model_name}")
    
    # Test Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "Hello, can you respond with a simple 'Hello, GraphRAG!' message?"
            }]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            print("✅ Gemini API test successful!")
            print(f"📝 Response: {text}")
            print(f"🎯 Use this model name in settings: {model_name.replace('models/', '')}")
            return True
        else:
            print(f"❌ Gemini API error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Gemini API connectivity...")
    success = test_gemini_api()
    
    if success:
        print("\n💡 Gemini API is working! GraphRAG configuration may need adjustment.")
        print("📋 Consider using OpenAI for embeddings and Gemini for chat models.")
    else:
        print("\n⚠️  Check your GOOGLE_API_KEY in .env file")