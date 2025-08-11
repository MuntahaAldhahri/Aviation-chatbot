import os
import time
from flask import Flask, request, jsonify, render_template, session
import openai
import requests
from uuid import uuid4

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", str(uuid4()))  # Required for session memory

# Load environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Configure Azure OpenAI
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


def run_search(query: str, query_type: str = "semantic"):
    """
    Call Azure Cognitive Search.
    We removed 'queryLanguage' for compatibility with your service.
    """
    search_url = (
        f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search"
        f"?api-version=2023-11-01"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY
    }
    payload = {
        "search": query,
        "top": 5,
        "queryType": query_type,                 # "semantic" or "simple"
        "answers": "extractive|count-3",
        "captions": "extractive|highlight-true",
        "searchFields": "content,title,filename",
        "select": "content,title,filename,metadata_storage_name",
        "semanticConfiguration": "default"       # remove if you don't have this config
    }
    resp = requests.post(search_url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get('message', '').strip()

    if not user_question:
        return jsonify({"answer": "Please enter a valid question."}), 400

    # Friendly greetings shortcut
    friendly_phrases = {"hi", "hello", "hey", "how are you", "good morning", "good evening", "what's up"}
    if user_question.lower() in friendly_phrases:
        return jsonify({"answer": "Hello! 👋 I'm your assistant. Ask me anything from the aviation documents!"})

    # Step 1: Azure Cognitive Search (semantic → simple fallback)
    try:
        results = run_search(user_question, "semantic")
        hits = results.get("value", [])

        if not hits or all(not (h.get("content") or "").strip() for h in hits):
            results = run_search(user_question, "simple")
            hits = results.get("value", [])

        # Build context from captions (best passages), fallback to content
        documents = []
        for h in hits:
            content = (h.get("content") or "").strip()
            captions = " ".join([c.get("text", "") for c in h.get("@search.captions", []) if c.get("text")])
            best_passage = captions if len(captions) > 60 else content
            if best_passage:
                documents.append(best_passage)

        context = ("\n\n---\n\n".join(documents))[:6000]  # max context size

        if not context:
            return jsonify({"answer": "❌ I couldn’t find anything related to your question in the provided aviation documents."})

        # Step 2: Maintain chat memory
        chat_history = session.get("chat_history", [])
        chat_history.append({"role": "user", "content": user_question})

        # Step 3: Compose Prompt
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

        # Step 4: Azure OpenAI ChatCompletion
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                response = openai.ChatCompletion.create(
                    engine=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=prompt,
                    temperature=0.2,
                    max_tokens=800
                )
                answer = response.choices[0].message['content']

                # Save assistant reply
                chat_history.append({"role": "assistant", "content": answer})
                session["chat_history"] = chat_history

                return jsonify({"answer": answer})

            except openai.error.RateLimitError:
                if attempt < retry_attempts - 1:
                    time.sleep(5)
                else:
                    return jsonify({"answer": "Rate limit exceeded. Please wait and try again."}), 429

    except requests.HTTPError as e:
        detail = e.response.text if getattr(e, "response", None) else str(e)
        return jsonify({"answer": f"Search API error: {detail}"}), 500
    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"answer": f"An internal error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
