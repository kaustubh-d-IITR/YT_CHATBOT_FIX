from groq import Groq
import os

def test_groq_connection():
    """Test Groq API connection"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment.")
        return False
    
    try:
        client = Groq(api_key=api_key)
        
        # Test with a simple question
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated model; previous llama3-8b-8192 deprecated
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you confirm you're working?"}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        print("✅ Groq API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Groq API connection failed: {e}")
        return False

if __name__ == "__main__":
    test_groq_connection()
