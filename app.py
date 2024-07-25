from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
best_model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']
    text_vectorized = vectorizer.transform([text])
    prediction = best_model.predict(text_vectorized)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)


