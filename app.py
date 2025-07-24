import os
import requests
from flask import Flask, request, jsonify, render_template
import openai

app = Flask(__name__)

# Environment variables (set in Azure App Settings)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Configure OpenAI SDK
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

    if not user_question:
        return jsonify({'answer': "Please enter a question."})

    # Step 1: Query Azure Cognitive Search
    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_SEARCH_API_KEY
    }
    search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2023-07-01-preview"
    payload = {
        "search": user_question,
        "top": 3,
        "select": "content"
    }

    try:
        response = requests.post(search_url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
        documents = results.get("value", [])
        if not documents:
            return jsonify({'answer': "Sorry, I couldn't find anything relevant in the company documents."})
    except Exception as e:
        print("Search error:", e)
        return jsonify({'answer': "Error connecting to server."})

    # Step 2: Prepare context from results
    combined_content = "\n".join([doc.get("content", "") for doc in documents])
    system_prompt = (
        "You are a helpful assistant for company aviation policies.\n"
        "ONLY answer using the content retrieved from Azure Cognitive Search.\n"
        "If you don't find the answer, reply: 'Sorry, I couldn't find anything relevant in the company documents.'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_question}\n\nRelevant info:\n{combined_content}"}
    ]

    try:
        completion = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.2
        )
        answer = completion.choices[0].message.content.strip()
        return jsonify({'answer': answer})

    except Exception as e:
        print("OpenAI error:", e)
        return jsonify({'answer': "Error connecting to server."})

if __name__ == '__main__':
    app.run(debug=True)
