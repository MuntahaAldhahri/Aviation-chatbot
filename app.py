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

# Set OpenAI configuration
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-04-14"
openai.api_key = AZURE_OPENAI_API_KEY

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    # Use Azure Cognitive Search to find documents
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
        response = requests.post(search_url, headers=headers, json=search_payload)
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        return jsonify({'answer': 'Error connecting to search service.', 'error': str(e)}), 500

    if 'value' not in results or len(results['value']) == 0:
        return jsonify({'answer': "Sorry, I couldn't find anything relevant in the company documents."})

    # Extract content from search results
    extracted_content = "\n\n".join([doc.get('content', '') for doc in results['value']])

    # Prepare messages
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant for company aviation policies. "
            "ONLY answer using the content retrieved from Azure Cognitive Search. "
            "If you don't find the answer, reply: 'Sorry, I couldn't find anything relevant in the company documents.'"
        )},
        {"role": "user", "content": f"Context:\n{extracted_content}\n\nQuestion: {user_question}"}
    ]

    try:
        completion = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.2
        )
        answer = completion['choices'][0]['message']['content'].strip()
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': 'Error connecting to server.', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
