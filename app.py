import os
from flask import Flask, render_template, request, jsonify
import requests
import openai
from dotenv import load_dotenv

# Load .env variables (optional, for local dev)
load_dotenv()

app = Flask(__name__)

# Load environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Configure OpenAI SDK (old SDK ≤0.28.1)
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2025-04-14"
openai.api_key = AZURE_OPENAI_API_KEY

@app.route('/')
def index():
    return "Aviation Chatbot is running!"  # For quick test, replace with render_template('index.html') if you have it.

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
        return jsonify({'answer': 'Server misconfiguration: Missing Azure OpenAI setup.'}), 500

    documents = ""
    if AZURE_SEARCH_API_KEY and AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX_NAME:
        search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2021-04-30-Preview"
        headers = {'Content-Type': 'application/json', 'api-key': AZURE_SEARCH_API_KEY}
        search_payload = {"search": user_question, "top": 3}
        try:
            search_response = requests.post(search_url, headers=headers, json=search_payload)
            search_response.raise_for_status()
            search_results = search_response.json()
            documents = "\n".join([doc.get('content', '') for doc in search_results.get('value', []) if doc.get('content')])
        except Exception as e:
            return jsonify({'answer': f'Cognitive Search error: {str(e)}'}), 500

    system_prompt = (
        "You are a helpful assistant for company aviation policies. "
        "ONLY answer using the content retrieved from Azure Cognitive Search. "
        "If you don’t find the answer, reply: 'Sorry, I couldn’t find anything relevant in the company documents.'"
    )
    full_prompt = f"Context:\n{documents}\n\nQuestion: {user_question}"

    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        answer = response['choices'][0]['message']['content']
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'answer': f'OpenAI error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
