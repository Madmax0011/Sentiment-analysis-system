
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the best model and vectorizer
best_model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    processed_text = vectorizer.transform([text])
    prediction = best_model.predict(processed_text)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
