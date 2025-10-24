import os
from flask import Flask, render_template, request
from huggingface_hub import InferenceClient

app = Flask(__name__)

client = InferenceClient(
    provider="nebius",
    api_key=os.environ.get("HF_TOKEN", "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"),
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    tone = request.form['tone']
    full_prompt = f"Write a {tone} {prompt}"
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
        )
        generated_text = completion.choices[0].message.content
    except Exception as e:
        generated_text = f'Error: {str(e)}'
    return render_template('index.html', result=generated_text, prompt=prompt, tone=tone)

if __name__ == '__main__':
    app.run(debug=True)
