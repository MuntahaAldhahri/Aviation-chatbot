@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    # Check configs
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
        return jsonify({'answer': 'Server misconfiguration: Missing Azure OpenAI setup.'}), 500

    # Prepare Cognitive Search (if set up)
    documents = ""
    if AZURE_SEARCH_API_KEY and AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX_NAME:
        search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2021-04-30-Preview"
        headers = {'Content-Type': 'application/json', 'api-key': AZURE_SEARCH_API_KEY}
        search_payload = {"search": user_question, "top": 3}
        try:
            search_response = requests.post(search_url, headers=headers, json=search_payload)
            search_response.raise_for_status()
            search_results = search_response.json()
            documents = "\n".join([doc.get('content', '') for doc in search_results.get('value', [])])
        except Exception as e:
            return jsonify({'answer': f'Search error: {str(e)}'}), 500

    # Compose prompt with strict instruction
    system_prompt = (
        "You are a helpful assistant for internal company policies. "
        "ONLY answer using the content retrieved from the Azure Cognitive Search index connected to this chat. "
        "If the answer cannot be found in the data, respond: "
        "\"Sorry, I couldn’t find anything relevant in the company documents.\""
    )

    prompt = f"Context:\n{documents}\n\nQuestion: {user_question}"

    # Ask Azure OpenAI
    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = "2024-04-14"
    openai.api_key = AZURE_OPENAI_API_KEY

    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        answer = response['choices'][0]['message']['content']
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'answer': f'OpenAI error: {str(e)}'}), 500
