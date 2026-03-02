import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Gemini API Key from Railway Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ===== Your Custom API Key Prefix =====
VALID_PREFIX = "An-animesh-official"

# ===== Home Route =====
@app.route("/")
def home():
    return "🔥 Andero A1.0 API is Running"

# ===== Chat Endpoint =====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_api_key = data.get("api_key")
    user_message = data.get("message")

    # Validate API Key
    if not user_api_key or not user_api_key.startswith(VALID_PREFIX):
        return jsonify({"error": "Invalid API Key"}), 403

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Gemini API URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": user_message}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        ai_reply = result["candidates"][0]["content"]["parts"][0]["text"]

        return jsonify({
            "model": "Andero-A1.0",
            "reply": ai_reply
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
