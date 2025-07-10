import os
from flask import Flask, render_template, request, jsonify
import openai
import requests

app = Flask(__name__)

# Load Azure keys from environment variables (set these in Azure App Service Configuration)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question', '')
    if not user_question:
        return jsonify({'answer': 'No question received.'}), 400

    openai.api_type = 'azure'
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_key = AZURE_OPENAI_API_KEY
    openai.api_version = '2023-05-15'  # check your Azure OpenAI API version

    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an assistant answering only from aviation documents."},
                {"role": "user", "content": user_question}
            ],
            temperature=0
        )
        answer = response['choices'][0]['message']['content'].strip()
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
