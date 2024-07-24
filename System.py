
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Preprocess Data
def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.ffill(inplace=True)  # Use forward fill to handle missing values
    data['Sentiment'] = data['Score'].apply(lambda x: 1 if x > 3 else 0)  # Binary sentiment
    return data

data = preprocess_data('amazon_fine_food_reviews.csv')
print("Data preprocessing completed.")

# Check the size of the dataset
print(f"Dataset size: {data.shape}")

# Split data
X = data['Text']
y = data['Sentiment']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2)
}

# Evaluate each model and print metrics
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_val_tfidf)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)

    print(f"Model: {model_name}")
    print(f"  Accuracy: {accuracy}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-Score: {f1}")

print("Model evaluation completed.")

# Used a smaller subset of the data (to check which model is taking considerable amount of time)
subset_data = data.sample(n=10000, random_state=42)
X = subset_data['Text']
y = subset_data['Sentiment']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Evaluate each model and print metrics
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_val_tfidf)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)

    print(f"Model: {model_name}")
    print(f"  Accuracy: {accuracy}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-Score: {f1}")

print("Model evaluation with subset data completed.")

