import os
from flask import Flask, render_template, request, jsonify
import openai

app = Flask(__name__)

# Azure config (use env variables or set directly — for now showing direct)
openai.api_type = "azure"
openai.api_base = "https://sans.openai.azure.com/"
openai.api_version = "2025-04-14"  
openai.api_key = "YOUR-REAL-API-KEY"

AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an assistant answering aviation questions only."},
                {"role": "user", "content": user_question}
            ],
            temperature=0
        )
        answer = response['choices'][0]['message']['content']
    except Exception as e:
        answer = f"Error: {str(e)}"

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
