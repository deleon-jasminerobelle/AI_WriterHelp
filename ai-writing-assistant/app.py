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

@app.route('/grammar-checker', methods=['GET', 'POST'])
def grammar_checker():
    if request.method == 'POST':
        text = request.form['text']
        try:
            # Use Hugging Face for grammar correction
            correction = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": f"Correct the grammar and improve the writing of the following text: {text}"
                    }
                ],
            )
            corrected_text = correction.choices[0].message.content
        except Exception as e:
            corrected_text = f'Error: {str(e)}'
        return render_template('grammar_checker.html', original=text, corrected=corrected_text)
    return render_template('grammar_checker.html')

@app.route('/templates', methods=['GET', 'POST'])
def templates():
    if request.method == 'POST':
        category = request.form['category']
        try:
            # Use Hugging Face to generate template
            template_request = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate a professional template for a {category}. Include placeholders and structure."
                    }
                ],
            )
            template = template_request.choices[0].message.content
        except Exception as e:
            template = f'Error: {str(e)}'
        return render_template('templates.html', category=category, template=template)
    return render_template('templates.html')

@app.route('/tips', methods=['GET', 'POST'])
def tips():
    if request.method == 'POST':
        topic = request.form['topic']
        try:
            # Use Hugging Face to generate writing tips
            tips_request = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": f"Provide detailed writing tips for: {topic}. Include practical advice and examples."
                    }
                ],
            )
            tips = tips_request.choices[0].message.content
        except Exception as e:
            tips = f'Error: {str(e)}'
        return render_template('tips.html', topic=topic, tips=tips)
    return render_template('tips.html')

@app.route('/editor', methods=['GET', 'POST'])
def editor():
    if request.method == 'POST':
        text = request.form['text']
        suggestion_type = request.form['suggestion_type']
        try:
            # Use Hugging Face for text suggestions
            suggestion_request = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": f"{suggestion_type.capitalize()} the following text: {text}"
                    }
                ],
            )
            suggestions = suggestion_request.choices[0].message.content
        except Exception as e:
            suggestions = f'Error: {str(e)}'
        return render_template('editor.html', text=text, suggestion_type=suggestion_type, suggestions=suggestions)
    return render_template('editor.html')

@app.route('/resources', methods=['GET', 'POST'])
def resources():
    if request.method == 'POST':
        query = request.form['query']
        try:
            # Use Hugging Face to find resources
            resources_request = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": f"Recommend writing resources for: {query}. Include books, websites, courses, and tools with brief descriptions."
                    }
                ],
            )
            resources = resources_request.choices[0].message.content
        except Exception as e:
            resources = f'Error: {str(e)}'
        return render_template('resources.html', query=query, resources=resources)
    return render_template('resources.html')

if __name__ == '__main__':
    app.run(debug=True)
