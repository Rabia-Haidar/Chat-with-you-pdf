import os
from dotenv import load_dotenv
import google.generativeai as genai

def list_available_models():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please ensure your .env file is correctly set up.")
        return

    genai.configure(api_key=api_key)

    print("Available Gemini Models:")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"- {m.name}")

if __name__ == "__main__":
    list_available_models()