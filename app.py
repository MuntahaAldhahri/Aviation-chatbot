import os
from flask import Flask, request, jsonify, render_template
import openai
import requests

app = Flask(__name__)

# Load environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Configure Azure OpenAI
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_API_KEY

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question', '').strip()

    try:
        # Friendly responses for general greetings
        friendly_phrases = ["hi", "hello", "hey", "how are you", "good morning", "good evening", "what's up"]
        if user_question.lower() in friendly_phrases:
            return jsonify({"answer": "Hello! 👋 I'm your assistant. Ask me anything from the aviation documents!"})

        # Step 1: Search documents using Azure Cognitive Search
        search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2023-07-01-Preview"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_SEARCH_API_KEY
        }
        search_payload = {
            "search": user_question,
            "top": 3
        }

        search_response = requests.post(search_url, headers=headers, json=search_payload)
        search_response.raise_for_status()
        results = search_response.json()
        documents = [doc.get("content", "") for doc in results.get("value", [])]
        context = "\n\n".join(documents)

        if not context.strip():
            return jsonify({"answer": "Sorry, I couldn’t find anything related to your question in the uploaded documents."})

        # Step 2: Send to Azure OpenAI
        response = openai.ChatCompletion.create(
            deployment_id=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and friendly assistant. Use only the context from the company's documents to answer. "
                        "If you don't find an answer in the context, respond that it is not available."
                    )
                },
                {
                    "role": "user",
                    "content": f"{user_question}\n\nContext:\n{context}"
                }
            ],
            temperature=0.5,
            max_tokens=800,
        )

        answer = response.choices[0].message['content']
        return jsonify({"answer": answer})

    except Exception as e:
        print("❌ ERROR during OpenAI call:", str(e))
        return jsonify({"answer": f"Error connecting to OpenAI service: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)
