@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    if not all([
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT_NAME,
        AZURE_SEARCH_API_KEY,
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_INDEX_NAME
    ]):
        return jsonify({"answer": "One or more environment variables are missing."})

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

        documents = [doc.get("content", "") for doc in search_results.get("value", [])]
        if not documents:
            return jsonify({"answer": "Hi! I couldn’t find anything related to that in the company documents. Try rephrasing or asking about a specific policy section."})

    except Exception as e:
        return jsonify({"answer": f"Error connecting to search service: {str(e)}"})

    # Ask OpenAI using only retrieved context
    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly and helpful assistant. Only answer based on the provided company documents. If the answer is not in the documents, say you couldn’t find anything relevant."
                },
                {
                    "role": "user",
                    "content": f"{user_question}\n\nContext:\n" + "\n---\n".join(documents)
                }
            ],
            temperature=0.6,
            max_tokens=800,
        )

        answer = response.choices[0].message['content']
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"Error connecting to OpenAI service: {str(e)}"})
