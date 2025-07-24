import os
from flask import Flask, render_template, request, jsonify
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

# Configure OpenAI for Azure
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-04-15-preview"
openai.api_key = AZURE_OPENAI_API_KEY

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
        return jsonify({"answer": "Azure OpenAI configuration is missing."})

    # Query Azure Cognitive Search
    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_SEARCH_API_KEY
    }
    search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2021-04-30-Preview"
    search_payload = {
        "search": user_question,
        "top": 3
    }
    try:
        search_response = requests.post(search_url, headers=headers, json=search_payload)
        search_response.raise_for_status()
        search_results = search_response.json()
        documents = [doc.get("content", "") for doc in search_results.get("value", [])]
        if not documents:
            return jsonify({"answer": "Sorry, I couldn't find anything relevant in the company documents."})
    except Exception as e:
        return jsonify({"answer": f"Error connecting to search service: {str(e)}"})

    # Ask Azure OpenAI
    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "Answer only based on the company documents provided."},
                {"role": "user", "content": f"{user_question}\n\nContext:\n" + "\n---\n".join(documents)}
            ],
            temperature=0.5,
            max_tokens=800,
        )
        answer = response.choices[0].message['content']
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error connecting to server: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
