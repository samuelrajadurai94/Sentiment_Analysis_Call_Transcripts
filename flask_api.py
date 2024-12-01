from flask import Flask, request, jsonify
from transformers import pipeline
import os
from nltk.tokenize import sent_tokenize  # For sentence tokenization
import nltk

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the sentiment analysis model
sentiment_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device=0)

@app.route('/upload', methods=['POST'])
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
    result = sentiment_pipeline(text, candidate_labels=labels)
    scores = dict(zip(result["labels"], result["scores"]))
    sentiment = max(scores, key=scores.get)
    return {
        "scores": scores,
        "sentiment": sentiment
    }
    

if __name__ == '__main__':
    app.run(debug=True)
