import os
import time
from uuid import uuid4
from typing import List, Tuple

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

# Optional: USE_SEMANTIC=true to enable semantic search (requires semantic config named "default")
USE_SEMANTIC = (os.getenv("USE_SEMANTIC", "false").strip().lower() == "true")

# Helpful warnings if any env is missing
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
    'select' includes fields that exist in both your current and cleaned schema.
    """
    select_fields = "content,title,filename,metadata_storage_name"
    if use_semantic:
        return {
            "search": query or "*",
            "top": 3,
            "queryType": "semantic",
            "semanticConfiguration": "default",
            "answers": "extractive|count-3",
            "captions": "extractive",
            # "queryLanguage": "en-us",  # optional
            "select": select_fields
        }
    return {
        "search": query or "*",
        "top": 3,
        "queryType": "simple",
        "select": select_fields
    }


def cognitive_search(query: str) -> Tuple[List[str], List[str]]:
    """
    Run a search query against the Azure Cognitive Search index.
    Returns (contents, sources) where sources are filenames for debugging/citation.
    """
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_INDEX_NAME or not AZURE_SEARCH_API_KEY:
        raise RuntimeError("Azure Search configuration is missing. Check your env vars.")

    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2023-11-01"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}

    # Try semantic first if enabled; auto-fallback to simple if it errors.
    payload = build_search_payload(query, use_semantic=USE_SEMANTIC)
    resp = requests.post(url, headers=headers, json=payload)
    if not resp.ok and USE_SEMANTIC:
        print("[INFO] Semantic search failed; falling back to simple. Error:", resp.text)
        payload = build_search_payload(query, use_semantic=False)
        resp = requests.post(url, headers=headers, json=payload)

    if not resp.ok:
        raise RuntimeError(f"Search API error {resp.status_code}: {resp.text}")

    data = resp.json() or {}
    hits = data.get("value", []) or []

    contents, sources = [], []
    for d in hits:
        txt = (d.get("content") or "").strip()
        # prefer 'title', then 'filename', then 'metadata_storage_name'
        name = d.get("title") or d.get("filename") or d.get("metadata_storage_name") or "document"
        if txt:
            contents.append(txt)
            sources.append(str(name))

    if not contents:
        print("[DEBUG] No hits for query:", query, "Raw response:", data)

    return contents, sources


def call_openai_with_context(user_question: str, context: str) -> str:
    """
    Send the prompt to Azure OpenAI using Chat Completions API.
    Enforces strict fallback when the answer isn't present in DOCUMENTS.
    """
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT_NAME:
        raise RuntimeError("Azure OpenAI configuration is missing. Check your env vars.")

    system_msg = (
        "You are a helpful assistant for aviation operations.\n"
        "You MUST base your answer ONLY on the text provided in the DOCUMENTS section.\n"
        "If the answer is not present in the DOCUMENTS, reply EXACTLY with:\n"
        "This information is not available in the provided documents.\n"
        "Do not add any other text when you use that sentence."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"QUESTION: {user_question}\n\nDOCUMENTS:\n{context}"}
    ]

    for attempt in range(3):
        try:
            res = openai.ChatCompletion.create(
                engine=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
            )
            return res.choices[0].message["content"]
        except openai.error.RateLimitError as e:
            if attempt < 2:
                time.sleep(5)
                continue
            raise e
        except Exception as e:
            raise e


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


@app.route("/debug-search")
def debug_search():
    """
    Quick endpoint to inspect what search returns.
    Usage: /debug-search?q=Annex
    """
    q = request.args.get("q", "Air Traffic")
    try:
        docs, sources = cognitive_search(q)
        return jsonify({"query": q, "hits": len(docs), "sources": sources, "preview": docs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_question = (data.get("message") or "").strip()
        if not user_question:
            return jsonify({"answer": "Please enter a valid question."}), 400

        # quick friendly responses
        friendly = {"hi", "hello", "hey", "how are you", "good morning", "good evening", "what's up"}
        if user_question.lower() in friendly:
            return jsonify({"answer": "Hello! 👋 I'm your assistant. Ask me anything from the aviation documents!"})

        # 1) Retrieve from Cognitive Search
        docs, sources = cognitive_search(user_question)
        if not docs:
            return jsonify({"answer": "❌ I couldn’t find anything related to your question in the provided aviation documents."})

        # Build a source-tagged context for transparency
        chunks = []
        for i, (text, src) in enumerate(zip(docs, sources), start=1):
            chunks.append(f"[SOURCE {i}: {src}]\n{text}")
        context = ("\n\n".join(chunks))[:3000]  # cap for token safety

        # 2) (optional) memory
        chat_history = session.get("chat_history", [])
        chat_history.append({"role": "user", "content": user_question})

        # 3) Generate answer
        answer = call_openai_with_context(user_question, context)

        # Append sources so you can see where it came from
        answer_with_sources = f"{answer}\n\nSources: " + ", ".join(f"[{i+1}] {s}" for i, s in enumerate(sources))
        chat_history.append({"role": "assistant", "content": answer_with_sources})
        session["chat_history"] = chat_history

        return jsonify({"answer": answer_with_sources})

    except openai.error.RateLimitError:
        return jsonify({"answer": "Rate limit exceeded. Please wait and try again."}), 429
    except Exception as e:
        print("❌ ERROR:", repr(e))
        return jsonify({"answer": f"An internal error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # Host/port can be overridden with PORT env
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
