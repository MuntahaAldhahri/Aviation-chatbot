import os
from flask import Flask, render_template, request, jsonify
import openai
import requests

app = Flask(__name__)

# Your Azure credentials (from screenshots)
AZURE_OPENAI_API_KEY = "0d3e7b5d2f49494a8673ce5f63d2b611"
AZURE_OPENAI_ENDPOINT = "https://sans.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"

AZURE_SEARCH_API_KEY = "14BB1579694F4BCBAA9914975F4E49F8"
AZURE_SEARCH_ENDPOINT = "https://sanssearchservice.search.windows.net"
AZURE_SEARCH_INDEX_NAME = "azureblob-index"

# Configure OpenAI for Azure
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_version = "2024-04-15"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    # Call Azure Cognitive Search
    search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY
    }
    search_body = {
        "search": user_question,
        "top": 3
    }

    try:
        search_response = requests.post(search_url, headers=headers, json=search_body)
        search_response.raise_for_status()
        results = search_response.json().get("value", [])

        if not results:
            return jsonify({"answer": "Sorry, I couldn't find anything relevant in the company documents."})

        content_chunks = "\n\n".join([doc.get("content", "") for doc in results])

        prompt = (
            "You are a helpful assistant for company aviation policies. "
            "ONLY answer using the content retrieved from Azure Cognitive Search. "
            "If you don't find the answer, reply: 'Sorry, I couldn't find anything relevant in the company documents.'\n\n"
            f"Context:\n{content_chunks}\n\nQuestion: {user_question}\nAnswer:"
        )

        chat_response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for company aviation policies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        final_answer = chat_response["choices"][0]["message"]["content"].strip()
        return jsonify({"answer": final_answer})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"answer": "Error connecting to server."})

if __name__ == '__main__':
    app.run(debug=True)
