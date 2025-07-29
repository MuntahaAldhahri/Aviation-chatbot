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
    user_question = request.json.get('question', '')

    try:
        print("🔍 DEBUG: OpenAI Call Preparation")
        print("🔧 Endpoint:", openai.api_base)
        print("🔧 Deployment (engine):", AZURE_OPENAI_DEPLOYMENT_NAME)
        print("🔧 API Version:", openai.api_version)
        print("🔧 API Key starts with:", AZURE_OPENAI_API_KEY[:5])

        # Step 1: Query Azure Cognitive Search
        search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2023-07-01-Preview"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_SEARCH_API_KEY
        }
        search_payload = {
            "search": user_question,
            "top": 5
        }

        search_response = requests.post(search_url, headers=headers, json=search_payload)
        search_response.raise_for_status()
        results = search_response.json()
        documents = [doc.get("content", "") for doc in results.get("value", [])]
        context = "\n\n".join(documents)

        if len(context) > 4000:
            context = context[:4000] + "\n\n...[truncated]..."

        # Step 2: Send to OpenAI
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and friendly assistant. "
                        "Only answer using the provided company documents. "
                        "If the answer is not in the documents, politely say so."
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
