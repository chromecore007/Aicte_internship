import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


data = pd.read_csv("dataset.csv")

X = data["clean_text"]
y = data["is_depression"]


# Convert text to numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("🔥 Model Accuracy:", accuracy)

# Save model & vectorizer
pickle.dump(model, open("mental_health_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Kaggle-trained model saved successfully")
