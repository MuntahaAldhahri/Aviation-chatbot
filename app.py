import os
import time
from uuid import uuid4

from flask import Flask, request, jsonify, render_template, session
import requests
import openai

# -------------------------
# Flask setup
# -------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", str(uuid4()))  # Required for session memory

# -------------------------
# Environment variables
# -------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = (os.getenv("AZURE_SEARCH_ENDPOINT") or "").rstrip("/")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Optional: USE_SEMANTIC=true to enable semantic search (requires a semantic config named "default")
USE_SEMANTIC = (os.getenv("USE_SEMANTIC", "false").strip().lower() == "true")

# Validate required envs early (flask will still start, but logs will show what's missing)
REQUIRED_ENVS = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
    "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
    "AZURE_SEARCH_API_KEY": AZURE_SEARCH_API_KEY,
    "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
    "AZURE_SEARCH_INDEX_NAME": AZURE_SEARCH_INDEX_NAME,
}
for k, v in REQUIRED_ENVS.items():
    if not v:
        print(f"[WARN] Missing env var: {k}")

# -------------------------
# OpenAI client configuration
# -------------------------
openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION


# -------------------------
# Helpers
# -------------------------
def build_search_payload(query: str, use_semantic: bool) -> dict:
    """
    Build a Cognitive Search request payload.
    Select fields are chosen to work with both your old index and the cleaned schema.
    """
    select_fields = "content,title,filename,metadata_storage_name"
    if use_semantic:
        return {
            "search": query,
            "top": 3,
            "queryType": "semantic",
            "semanticConfiguration": "default",
            "answers": "extractive|count-3",
            "captions": "extractive",
            # "queryLanguage": "en-us",  # optional
            "select": select_fields
        }
    else:
        return {
            "search": query or "*",
            "top": 3,
            "queryType": "simple",
            "select": select_fields
        }


def cognitive_search(query: str) -> list[str]:
    """
    Run a search query against the Azure Cognitive Search index.
    Returns a list of document text chunks (strings).
    Falls back to simple search automatically if semantic search errors.
    """
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_INDEX_NAME or not AZURE_SEARCH_API_KEY:
        raise RuntimeError("Azure Search configuration is missing. Check your env vars.")

    url = (
        f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search"
        f"?api-version=2023-11-01"
    )
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}

    # 1) Try semantic if enabled
    payload = build_search_payload(query, use_semantic=USE_SEMANTIC)
    resp = requests.post(url, headers=headers, json=payload)
    if not resp.ok and USE_SEMANTIC:
        # automatic fallback: try simple query once
        print("[INFO] Semantic search failed, falling back to simple. Error:", resp.text)
        payload = build_search_payload(query, use_semantic=False)
        resp = requests.post(url, headers=headers, json=payload)

    if not resp.ok:
        raise RuntimeError(f"Search API error {resp.status_code}: {resp.text}")

    data = resp.json() or {}
    hits = data.get("value", []) or []

    docs: list[str] = []
    for d in hits:
        # Prefer 'content'; fall back to nothing if absent
        txt = d.get("content") or ""
        if isinstance(txt, str) and txt.strip():
            docs.append(txt.strip())

    return docs


def call_openai(user_question: str, context: str) -> str:
    """
    Send the prompt to Azure OpenAI using Chat Completions API.
    """
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT_NAME:
        raise RuntimeError("Azure OpenAI configuration is missing. Check your env vars.")

    system_msg = (
        "You are a helpful assistant for aviation operations. "
        "ONLY answer using the information provided in the DOCUMENTS section. "
        "If the answer is not found, reply with: "
        "'This information is not available in the provided documents.'"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"QUESTION: {user_question}\n\nDOCUMENTS:\n{context}"}
    ]

    # simple retry for rate limits
    for attempt in range(3):
        try:
            res = openai.ChatCompletion.create(
                engine=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
            )
            return res.choices[0].message["content"]
        except openai.error.RateLimitError:
            if attempt < 2:
                time.sleep(5)
            else:
                raise
        except Exception:
            raise


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def health():
    return "OK", 200


@app.route("/reset", methods=["POST"])
def reset_memory():
    session.pop("chat_history", None)
    return jsonify({"message": "Chat history cleared."})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_question = (data.get("message") or "").strip()
        if not user_question:
            return jsonify({"answer": "Please enter a valid question."}), 400

        # Quick friendly responses
        friendly = {"hi", "hello", "hey", "how are you", "good morning", "good evening", "what's up"}
        if user_question.lower() in friendly:
            return jsonify({"answer": "Hello! 👋 I'm your assistant. Ask me anything from the aviation documents!"})

        # 1) Retrieve from Cognitive Search
        docs = cognitive_search(user_question)
        context = "\n\n".join(docs)[:3000]  # cap context length for token safety

        if not context:
            return jsonify({"answer": "❌ I couldn’t find anything related to your question in the provided aviation documents."})

        # 2) Update memory (optional)
        chat_history = session.get("chat_history", [])
        chat_history.append({"role": "user", "content": user_question})

        # 3) Generate answer with Azure OpenAI
        answer = call_openai(user_question, context)

        chat_history.append({"role": "assistant", "content": answer})
        session["chat_history"] = chat_history

        return jsonify({"answer": answer})

    except openai.error.RateLimitError:
        return jsonify({"answer": "Rate limit exceeded. Please wait and try again."}), 429
    except Exception as e:
        print("❌ ERROR:", repr(e))
        return jsonify({"answer": f"An internal error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # Host/port can be set via env, e.g. PORT=8000
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
    
