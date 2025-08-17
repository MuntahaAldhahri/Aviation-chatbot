import os
import time
from uuid import uuid4
from typing import List, Tuple, Dict, Any

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
FALLBACK_SENTENCE = "This information is not available in the provided documents."

def build_search_payload(query: str, use_semantic: bool) -> dict:
    """
    Build a Cognitive Search request payload.
    'select' includes fields that exist in both your current and cleaned schema.
    """
    # Include metadata_storage_path so we can de-duplicate by file
    select_fields = "content,title,filename,metadata_storage_name,metadata_storage_path"

    if use_semantic:
        return {
            "search": query or "*",
            "top": 8,  # get enough to dedupe down to ~3 uniques
            "queryType": "semantic",
            "semanticConfiguration": "default",
            "answers": "extractive|count-3",
            "captions": "extractive",
            "select": select_fields
        }

    # Simple keyword search with highlights to get good snippets
    return {
        "search": query or "*",
        "top": 8,
        "queryType": "simple",
        "highlight": "content",
        "highlightPreTag": "<em>",
        "highlightPostTag": "</em>",
        "select": select_fields
    }


def _pick_name(d: Dict[str, Any]) -> str:
    return d.get("title") or d.get("filename") or d.get("metadata_storage_name") or "document"


def _pick_snippet(d: Dict[str, Any]) -> str:
    # Prefer semantic captions when present
    caps = d.get("@search.captions") or []
    if caps and isinstance(caps, list) and caps[0].get("text"):
        return caps[0]["text"]

    # Otherwise use highlights if present (simple mode)
    hl = d.get("@search.highlights", {})
    if isinstance(hl, dict):
        arr = hl.get("content")
        if arr and isinstance(arr, list):
            return arr[0]

    # Fallback to raw content (trim)
    txt = (d.get("content") or "").strip()
    return txt[:500]


def cognitive_search(query: str) -> Tuple[List[str], List[str]]:
    """
    Query Azure Cognitive Search and return (context_chunks, unique_source_names).
    - De-duplicates hits by metadata_storage_path (file).
    - Returns at most 3 unique documents.
    - Each chunk is tagged with [SOURCE i: name] for prompting.
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

    # Deduplicate by file (metadata_storage_path)
    seen_ids = set()
    chunks: List[str] = []
    sources: List[str] = []

    for d in hits:
        file_id = d.get("metadata_storage_path") or _pick_name(d)
        if file_id in seen_ids:
            continue
        seen_ids.add(file_id)

        name = _pick_name(d)
        snippet = _pick_snippet(d)
        if not snippet:
            continue

        chunks.append(f"[SOURCE {len(chunks)+1}: {name}]\n{snippet}")
        sources.append(str(name))

        if len(chunks) >= 3:
            break

    if not chunks:
        print("[DEBUG] No hits for query:", query, "Raw response:", data)

    return chunks, sources


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
        f"{FALLBACK_SENTENCE}\n"
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
    Quick endpoint to inspect what search returns (deduped).
    Usage: /debug-search?q=Annex
    """
    q = request.args.get("q", "Air Traffic")
    try:
        chunks, sources = cognitive_search(q)
        return jsonify({"query": q, "hits": len(chunks), "sources": sources, "preview": chunks})
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

        # 1) Retrieve from Cognitive Search (deduped, with snippets)
        chunks, sources = cognitive_search(user_question)
        if not chunks:
            return jsonify({"answer": "❌ I couldn’t find anything related to your question in the provided aviation documents."})

        context = ("\n\n".join(chunks))[:3000]  # cap for token safety

        # 2) (optional) memory
        chat_history = session.get("chat_history", [])
        chat_history.append({"role": "user", "content": user_question})

        # 3) Generate answer
        answer = call_openai_with_context(user_question, context)

        # If we got the exact fallback, don't add sources (keeps it clean)
        if answer.strip() == FALLBACK_SENTENCE:
            chat_history.append({"role": "assistant", "content": answer})
            session["chat_history"] = chat_history
            return jsonify({"answer": answer})

        # Otherwise, append deduped sources
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
