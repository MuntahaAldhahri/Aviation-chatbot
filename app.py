import os
from flask import Flask, request, jsonify, render_template
import openai
import requests

app = Flask(__name__)

# Load environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Configure Azure OpenAI
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-04-15-preview"
openai.api_key = AZURE_OPENAI_API_KEY

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question', '')

    # Check that all config is set
    if not all([
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT_NAME,
        AZURE_SEARCH_API_KEY,
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_INDEX_NAME
    ]):
        return jsonify({"answer": "Server configuration error. Please check environment variables."})

    # === Cognitive Search ===
    try:
        search_headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_SEARCH_API_KEY
        }

        search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2021-04-30-Preview"
        search_payload = {
            "search": user_question,
            "top": 3,
            "queryType": "semantic",
            "semanticConfiguration": "default",
            "queryLanguage": "en-us"
        }

        search_response = requests.post(search_url, headers=search_headers, json=search_payload)
        search_response.raise_for_status()
        search_results = search_response.json()

        # Log results for debugging
        print("Search Results:", search_results)

        docs = [doc.get("content", "") for doc in search_results.get("value", [])]

        if not docs:
            return jsonify({"answer": "Hi! I couldn’t find anything related to that in the company documents. Try rephrasing or asking about a specific policy."})

    except Exception as e:
        return jsonify({"answer": f"Error connecting to search service: {str(e)}"})

    # === Call Azure OpenAI ===
    try:
        context = "\n---\n".join(docs)

        # Limit context size (optional safety)
        if len(context) > 4000:
            context = context[:4000] + "\n\n...[truncated]..."

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
        return jsonify({"answer": f"Error connecting to OpenAI service: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
