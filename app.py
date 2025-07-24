import os
from flask import Flask, request, jsonify, render_template
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)

# Load environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Initialize OpenAI client
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-04-14",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# Initialize Azure Cognitive Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get("question", "")

    try:
        search_results = search_client.search(user_question, top=3)
        sources = []

        for result in search_results:
            content = result.get("content", "")
            filename = result.get("filename", "Document")
            sources.append(f"{filename}:\n{content}")

        if not sources:
            return jsonify({"answer": "Sorry, I couldn't find anything relevant in the company documents."})

        # Combine search snippets into one prompt
        context = "\n\n".join(sources)
        prompt = f"""Use the following company documents to answer the question.
Documents:
{context}

Question: {user_question}
Answer:"""

        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"answer": "Error connecting to server."})

if __name__ == "__main__":
    app.run(debug=True)
