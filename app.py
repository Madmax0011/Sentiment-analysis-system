import joblib
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Ensure the model and vectorizer files are correctly loaded
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

best_model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST' and 'text' in request.form:
        text = request.form['text']
        if text:
            prediction = make_prediction(text)
            sentiment = 'Positive' if prediction == 1 else 'Negative'
    return render_template('index.html', sentiment=sentiment)

def make_prediction(text):
    processed_text = vectorizer.transform([text])
    prediction = best_model.predict(processed_text)
    return prediction[0]

if __name__ == "__main__":
    app.run(debug=True)
