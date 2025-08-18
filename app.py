import os
import time
from flask import Flask, request, jsonify, render_template, session
import openai
import requests
from uuid import uuid4

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", str(uuid4()))  # Required for session memory

# ---- Env vars ----
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Optional: enable semantic after creating config "default" in the index
USE_SEMANTIC = os.getenv("USE_SEMANTIC", "false").lower() == "true"

# ---- OpenAI config ----
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_API_KEY


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/healthz')
def health():
    return "OK", 200


@app.route('/reset', methods=['POST'])
def reset_memory():
    session.pop("chat_history", None)
    return jsonify({"message": "Chat history cleared."})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = (data.get('message') or "").strip()

    if not user_question:
        return jsonify({"answer": "Please enter a valid question."}), 400

    # Friendly greetings shortcut
    friendly_phrases = {"hi", "hello", "hey", "how are you", "good morning", "good evening", "what's up"}
    if user_question.lower() in friendly_phrases:
        return jsonify({"answer": "Hello! 👋 I'm your assistant. Ask me anything from the aviation documents!"})

    # ---- Step 1: Azure Cognitive Search ----
    try:
        search_url = (
            f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search"
            f"?api-version=2023-11-01"
        )
        headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}

        if USE_SEMANTIC:
            # Only use this after you add a semantic config named "default" on the index
            search_payload = {
                "search": user_question,
                "top": 3,
                "queryType": "semantic",
                "queryLanguage": "en-us",
                "semanticConfiguration": "default",
                "answers": "extractive|count-3",
                "captions": "extractive",
                "select": "content,metadata_storage_name"
            }
        else:
            # Always-valid simple keyword search
            search_payload = {
                "search": user_question if user_question else "*",
                "top": 3,
                "queryType": "simple",
                "select": "content,metadata_storage_name"
            }

        search_response = requests.post(search_url, headers=headers, json=search_payload)
        if not search_response.ok:
            return jsonify({
                "answer": f"Search API error {search_response.status_code}: {search_response.text}"
            }), 500

        search_json = search_response.json()
        docs = search_json.get("value", [])
        documents = []
        for d in docs:
            # 'content' must exist in your index schema and be retrievable
            content = d.get("content") or ""
            if content and isinstance(content, str):
                documents.append(content)

        context = "\n\n".join(doc.strip() for doc in documents if doc.strip())[:3000]

        if not context:
            return jsonify({"answer": "❌ I couldn’t find anything related to your question in the provided aviation documents."})

        # ---- Step 2: Maintain chat memory ----
        chat_history = session.get("chat_history", [])
        chat_history.append({"role": "user", "content": user_question})

        # ---- Step 3: Compose Prompt ----
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for aviation operations. "
                    "ONLY answer using the information provided in the DOCUMENTS section. "
                    "If the answer is not found, reply with: 'This information is not available in the provided documents.'"
                )
            },
            {
                "role": "user",
                "content": f"QUESTION: {user_question}\n\nDOCUMENTS:\n{context}"
            }
        ]

        # ---- Step 4: Azure OpenAI ChatCompletion ----
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                response = openai.ChatCompletion.create(
                    engine=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=prompt,
                    temperature=0.2,
                    max_tokens=800
                )
                answer = response.choices[0].message["content"]
                chat_history.append({"role": "assistant", "content": answer})
                session["chat_history"] = chat_history
                return jsonify({"answer": answer})
            except openai.error.RateLimitError:
                if attempt < retry_attempts - 1:
                    time.sleep(5)
                else:
                    return jsonify({"answer": "Rate limit exceeded. Please wait and try again."}), 429

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"answer": f"An internal error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
