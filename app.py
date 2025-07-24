import os
from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI

app = Flask(__name__)

# Load environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # gpt-4.1
AZURE_OPENAI_API_VERSION = "2025-04-14"

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for company aviation policies.\n"
                        "ONLY answer using the content retrieved from Azure Cognitive Search.\n"
                        "If you don't find the answer, reply: 'Sorry, I couldn't find anything relevant in the company documents.'"
                    )
                },
                {"role": "user", "content": user_question}
            ],
            temperature=0.3,
            max_tokens=800
        )

        answer = response.choices[0].message.content
        return jsonify({'response': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
