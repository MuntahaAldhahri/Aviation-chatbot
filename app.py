import os
from flask import Flask, render_template, request, jsonify
import openai
import requests

app = Flask(__name__)

# Azure environment variables (set these in Azure App Service → Configuration)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    # Step 1: Query Azure Cognitive Search
    search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2021-04-30-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY
    }
    search_payload = {
        "search": user_question,
        "top": 3
    }

    try:
        search_response = requests.post(search_url, headers=headers, json=search_payload)
        search_response.raise_for_status()
        search_results = search_response.json()
    except Exception as e:
        return jsonify({"answer": f"Search error: {str(e)}"})

    # Collect documents
    documents = "\n".join([doc.get('content', '') for doc in search_results.get('value', [])])

    # Step 2: Ask Azure OpenAI using the context
    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = "2025-04-14"
    openai.api_key = AZURE_OPENAI_API_KEY

    prompt = f"Answer the following question ONLY using the provided context.\n\nContext:\n{documents}\n\nQuestion: {user_question}"

    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an assistant that only answers using company documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        answer = response['choices'][0]['message']['content']
    except Exception as e:
        return jsonify({"answer": f"OpenAI error: {str(e)}"})

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
