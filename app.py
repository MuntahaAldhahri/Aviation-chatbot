import os
import time
import json
from uuid import uuid4

from flask import Flask, request, jsonify, render_template, session
import requests
import openai

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", str(uuid4()))  # Required for session memory

# ---------- Env ----------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")  
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# ---------- Azure OpenAI (legacy openai lib) ----------
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
    Calls Azure Cognitive Search.
    query_type: 'semantic' (preferred) or 'simple' (fallback)
    """
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2023-11-01"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}

    payload = {
        "search": query,
        "top": 5,
        "queryType": query_type,
        "queryLanguage": "en-us",
        "answers": "extractive|count-3",
        "captions": "extractive|highlight-true",
        "searchFields": "content,title,filename",
        "select": "content,title,filename,metadata_storage_name"
    }
    # Keep if you created a semantic config named 'default'. If not, comment this line.
    payload["semanticConfiguration"] = "default"

    r = requests.post(url, headers=headers, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = (data.get('message') or '').strip()

    if not user_question:
        return jsonify({"answer": "Please enter a valid question."}), 400

    # Friendly greetings shortcut
    friendly_phrases = {"hi", "hello", "hey", "how are you", "good morning", "good evening", "what's up"}
    if user_question.lower() in friendly_phrases:
        return jsonify({"answer": "Hello! 👋 I'm your assistant. Ask me anything from the aviation documents!"})

    try:
        # --- 1) Semantic search ---
        results = run_search(user_question, "semantic")
        hits = results.get("value", [])

        # --- 2) Fallback to simple if semantic returns nothing/blank content ---
        if not hits or all(not (h.get("content") or "").strip() for h in hits):
            results = run_search(user_question, "simple")
            hits = results.get("value", [])

        # --- 3) Build context from captions (best passages), then content ---
        docs = []
        debug_meta = []
        for h in hits:
            content = (h.get("content") or "").strip()
            captions = " ".join([c.get("text", "") for c in h.get("@search.captions", []) if c.get("text")])
            best = captions if len(captions) > 60 else content
            if best:
                docs.append(best)
            debug_meta.append({
                "title": h.get("title"),
                "filename": h.get("filename") or h.get("metadata_storage_name"),
                "snippet": (best[:220] + "…") if best else ""
            })

        context = ("\n\n---\n\n".join(docs))[:6000]

        if not context:
            # This means retrieval failed (indexing/OCR/skillset/chunking/filters)
            return jsonify({
                "answer": ("❌ I couldn’t retrieve matching text from the indexed documents. "
                           "This likely indicates an indexing/skillset issue (e.g., missing OCR or poor extraction)."),
                "debug": {"hits": debug_meta}
            })

        # --- 4) Maintain chat memory (optional) ---
        chat_history = session.get("chat_history", [])
        chat_history.append({"role": "user", "content": user_question})

        # --- 5) Strict, grounded prompt ---
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for aviation operations. "
                    "Answer ONLY using the information provided in the DOCUMENTS section. "
                    "If the answer is not found, reply exactly: "
                    "'This information is not available in the provided documents.'"
                )
            },
            {
                "role": "user",
                "content": f"QUESTION: {user_question}\n\nDOCUMENTS:\n{context}"
            }
        ]

        # --- 6) Azure OpenAI call with retries ---
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                response = openai.ChatCompletion.create(
                    engine=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=prompt,
                    temperature=0.1,
                    max_tokens=800
                )
                answer = response.choices[0].message['content']

                chat_history.append({"role": "assistant", "content": answer})
                session["chat_history"] = chat_history

                return jsonify({"answer": answer, "debug": {"hits": debug_meta}})
            except openai.error.RateLimitError:
                if attempt < retry_attempts - 1:
                    time.sleep(5)
                else:
                    return jsonify({"answer": "Rate limit exceeded. Please wait and try again."}), 429

    except requests.HTTPError as e:
        # Surface Search API errors clearly
        try:
            detail = e.response.text
        except Exception:
            detail = str(e)
        return jsonify({"answer": f"Search API error: {detail}"}), 500
    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"answer": f"An internal error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
