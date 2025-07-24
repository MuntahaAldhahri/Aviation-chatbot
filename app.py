import os
from flask import Flask, render_template, request, jsonify
import openai
import requests

app = Flask(__name__)

# Load environment variables (already configured in Azure App Service)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = "https://sanssearchservice.search.windows.net"
AZURE_SEARCH_INDEX_NAME = "azureblob-index"

# Configure OpenAI
openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-04-14"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    
    # Step 1: Retrieve documents from Azure Cognitive Search
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
        response = requests.post(search_url, headers=headers, json=search_payload)
        response.raise_for_status()
        documents = response.json().get('value', [])
    except Exception as e:
        return jsonify({'answer': f"Error connecting to server: {str(e)}"})

    if not documents:
        return jsonify({'answer': "Sorry, I couldn't find anything relevant in the company documents."})

    context = "\n\n".join([doc.get('content', '') for doc in documents])

    # Step 2: Ask OpenAI using the context
    system_prompt = (
        "You are a helpful assistant for company aviation policies.\n"
        "ONLY answer using the content retrieved from Azure Cognitive Search.\n"
        "If you don't find the answer, reply: 'Sorry, I couldn't find anything relevant in the company documents.'"
    )

    try:
        completion = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
            ]
        )
        answer = completion.choices[0].message['content']
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': f"Error connecting to server: {str(e)}"})

if __name__ == '__main__':
    app.run()
