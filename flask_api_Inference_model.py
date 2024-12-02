from flask import Flask, request, jsonify
import os
import requests


app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


@app.route("/", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        with open(filepath, 'r') as f:
            content = f.read()

        sentiment = analyze_sentiment(content)
        return jsonify(sentiment)


def analyze_sentiment(text):
    labels = ["positive", "neutral", "negative"]
    result = query({"inputs": text,"parameters": {"candidate_labels": labels}})
    scores = dict(zip(result["labels"], result["scores"]))
    sentiment = max(scores, key=scores.get)
    return {
        "scores": scores,
        "sentiment": sentiment
    }
    

if __name__ == '__main__':
    app.run(debug=True)
