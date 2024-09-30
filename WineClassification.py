import pandas as pd
import pickle
import logging
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk import FreqDist
from scipy.sparse import hstack
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess training data
def load_data(train_path):
    logging.info("Loading training data...")
    df = pd.read_csv(train_path)
    df = df.dropna()
    logging.info("Data loaded successfully.")
    return df

# Combine review title and description
def process_reviews(df):
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review_title'] + ' ' + df['review_description']
    df['review'] = df['review'].apply(
        lambda x: ' '.join([w for w in word_tokenize(x.lower()) if w.isalpha() and w not in stop_words])
    )
    return df

# Vectorize text and add additional features
def vectorize_and_add_features(inputs, vectorizer):
    logging.info("Vectorizing reviews and adding features...")
    X_train_vectorized = vectorizer.transform(inputs['review'])
    X_train_vectorized_added = hstack([
        X_train_vectorized,
        inputs[['points', 'price']].values
    ])
    logging.info("Vectorization and feature addition completed.")
    return X_train_vectorized_added

# Train the model
def train_model(XTrain, YTrain):
    logging.info("Training the model...")
    model = RandomForestClassifier(n_estimators=250)
    model.fit(XTrain, YTrain)
    logging.info("Model training completed.")
    return model

# Save the model
def save_model(model, filename='wine_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved as {filename}")

# Load the saved model
def load_model(filename='wine_model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    logging.info(f"Model loaded from {filename}")
    return model

# NLP Insight 1: Most frequent word in reviews
def most_frequent_word(df):
    all_words = ' '.join(df['review']).split()
    freq_dist = FreqDist(all_words)
    most_common_word = freq_dist.most_common(1)[0][0]
    print(f"Most common word: {most_common_word}")
    return most_common_word

# NLP Insight 2: Longest review
def longest_review(df):
    longest_review_idx = df['review'].apply(len).idxmax()
    longest_review_text = df['review'].iloc[longest_review_idx]
    print(f"Longest review: {longest_review_text}")
    return longest_review_text

# NLP Insight 3: Average length of reviews
def average_review_length(df):
    avg_review_length = df['review'].apply(len).mean()
    print(f"Average review length: {avg_review_length:.2f} characters")
    return avg_review_length

# NLP Insight 4: Shortest review
def shortest_review(df):
    shortest_review_idx = df['review'].apply(len).idxmin()
    shortest_review_text = df['review'].iloc[shortest_review_idx]
    print(f"Shortest review: {shortest_review_text}")
    return shortest_review_text

# NLP Insight 5: Most common adjective in reviews
def most_common_adjective(df):
    all_words = ' '.join(df['review']).split()
    tagged_words = nltk.pos_tag(all_words)
    adjectives = [word for word, pos in tagged_words if pos == 'JJ']
    most_common_adjective = FreqDist(adjectives).most_common(1)[0][0]
    print(f"Most common adjective: {most_common_adjective}")
    return most_common_adjective

# NLP Insight 6: Most frequent review title
def most_frequent_review_title(df):
    most_common_title = df['review_title'].value_counts().idxmax()
    print(f"Most common review title: {most_common_title}")
    return most_common_title

# NLP Insight 7: Average points of wines
def average_points(df):
    avg_points = df['points'].mean()
    print(f"Average points of wines: {avg_points:.2f}")
    return avg_points

# NLP Insight 8: Most frequent country in the dataset
def most_frequent_country(df):
    most_common_country = df['country'].value_counts().idxmax()
    print(f"Most frequent country: {most_common_country}")
    return most_common_country

# Main function to execute the entire flow
def main(train_path, test_path):
    df_train = load_data(train_path)
    df_test = load_data(test_path)

    # Process reviews for both train and test datasets
    df_train = process_reviews(df_train)
    df_test = process_reviews(df_test)

    # Encode the target variable
    le_variety = LabelEncoder()
    df_train['variety_encoded'] = le_variety.fit_transform(df_train['variety'])

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    vectorizer.fit(df_train['review'])

    # Prepare features for training and testing
    X_train = vectorize_and_add_features(df_train, vectorizer)
    Y_train = df_train['variety_encoded']

    # Train the model
    model = train_model(X_train, Y_train)
    save_model(model)

    # Perform NLP insights
    most_frequent_word(df_train)
    longest_review(df_train)
    average_review_length(df_train)
    shortest_review(df_train)
    most_common_adjective(df_train)
    most_frequent_review_title(df_train)
    average_points(df_train)
    most_frequent_country(df_train)

if __name__ == "__main__":
    main('train500.csv', 'test500.csv')
