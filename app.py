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

    # Check for required configuration
    if not all([
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT_NAME,
        AZURE_SEARCH_API_KEY,
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_INDEX_NAME
    ]):
        return jsonify({"answer": "One or more environment variables are missing."})

    # Prepare Azure Cognitive Search query
    headers = {
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

    try:
        search_response = requests.post(search_url, headers=headers, json=search_payload)
        search_response.raise_for_status()
        search_results = search_response.json()

        # Log search results for debugging
        print("Search Results:", search_results)

        documents = []
        for doc in search_results.get("value", []):
            print("Matched Document:", doc)
            documents.append(doc.get("content", ""))  # Update field name if different

        if not documents:
            return jsonify({"answer": "Sorry, I couldn't find anything relevant in the company documents."})

    except Exception as e:
        return jsonify({"answer": f"Error connecting to search service: {str(e)}"})

    # Prepare prompt for OpenAI
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer only using the information provided in the context below. If the answer is not found in the context, reply: 'Sorry, I couldn't find anything relevant in the company documents.'"
            },
            {
                "role": "user",
                "content": f"{user_question}\n\nContext:\n" + "\n---\n".join(documents)
            }
        ]

        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )

        answer = response.choices[0].message['content']
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"Error connecting to OpenAI service: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
