#!/usr/bin/env python3
"""
Test LiteLLM with Gemini configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_litellm_gemini():
    """Test LiteLLM with Gemini"""
    
    try:
        import litellm
        
        # Set API key
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        
        # Test different model formats
        model_formats = [
            "gemini/gemini-2.5-flash",
            "vertex_ai/gemini-2.5-flash", 
            "google/gemini-2.5-flash",
            "gemini-2.5-flash"
        ]
        
        for model_format in model_formats:
            print(f"\n🧪 Testing model format: {model_format}")
            
            try:
                response = litellm.completion(
                    model=model_format,
                    messages=[{"role": "user", "content": "Hello, respond with just 'Hello GraphRAG!'"}],
                    max_tokens=20
                )
                
                print(f"✅ Success with {model_format}")
                print(f"📝 Response: {response.choices[0].message.content}")
                print(f"🎯 Use this format in GraphRAG: {model_format}")
                return model_format
                
            except Exception as e:
                print(f"❌ Failed with {model_format}: {str(e)[:100]}...")
                
        return None
        
    except ImportError:
        print("❌ LiteLLM not installed")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Testing LiteLLM with Gemini...")
    successful_format = test_litellm_gemini()
    
    if successful_format:
        print(f"\n✅ Success! Use this in settings.yaml:")
        print(f"model: {successful_format}")
    else:
        print(f"\n⚠️  All formats failed. Check your API quota and key.")