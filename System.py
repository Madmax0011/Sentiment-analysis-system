
import pandas as pd

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
