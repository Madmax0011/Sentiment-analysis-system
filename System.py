import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocess Data
def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.ffill(inplace=True)  # Use forward fill to handle missing values
    data['Sentiment'] = data['Score'].apply(lambda x: 1 if x > 3 else 0)  # Binary sentiment
    return data

data = preprocess_data('amazon_fine_food_reviews.csv')
# Data preprocessing check
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

# Evaluate model performance (accuracy, precision, recall, and F1-score)
def evaluate_model(model, X_train_tfidf, y_train, X_val_tfidf, y_val):
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    return accuracy, precision, recall, f1

# Evaluate each model and print metrics
performance_metrics = []

for model_name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train_tfidf, y_train, X_val_tfidf, y_val)
    performance_metrics.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    print(f"Model: {model_name}")
    print(f"  Accuracy: {accuracy}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1-Score: {f1}")

# Model evaluation check
print("Model evaluation with optimized Random Forest completed.")

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [10, 50],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# Perform grid search
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_tfidf, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model
best_accuracy, best_precision, best_recall, best_f1 = evaluate_model(best_model, X_train_tfidf, y_train, X_val_tfidf, y_val)

print(f"Best Model (Random Forest) Accuracy: {best_accuracy}")
print(f"Best Model (Random Forest) Precision: {best_precision}")
print(f"Best Model (Random Forest) Recall: {best_recall}")
print(f"Best Model (Random Forest) F1-Score: {best_f1}")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model optimization and evaluation completed.")

# DataFrame for performance metrics
performance_df = pd.DataFrame(performance_metrics)

# Setting up the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='value', hue='variable', data=performance_df.melt(id_vars='Model'))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.legend(title='Metric')
plt.show()
