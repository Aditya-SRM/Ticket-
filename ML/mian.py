import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob  # Library for sentiment analysis

# Define functions for preprocessing text, creating priority rules, and handling basic suggestions
def preprocess_text(text):
    text = text.lower()
    # Add more steps like punctuation removal, stemming/lemmatization
    return text

def define_priority_rules(urgency_words, critical_words):
    def prioritize(text):
        if any(word in text for word in urgency_words):
            return "High"
        elif any(word in text for word in critical_words):
            return "Critical"
        else:
            return "Medium"
    return prioritize

def suggest_faq(category, keywords):
    # Implement logic to retrieve a pre-defined FAQ response based on category and keywords
    # This is a placeholder for illustration
    faq_response = f"Here's a general response for category: {category}"
    return faq_response

# Load data
data = pd.read_csv("customer_support_tickets(1).csv")

# Preprocess text data
data["Ticket_Priority_Clean"] = data["Ticket Priority"].apply(preprocess_text)
data["Ticket_Description_Clean"] = data["Ticket Description"].apply(preprocess_text)

# Prepare features and target variable
X_text = data[["Ticket priority"]]
X_numeric = data[["Customer Age"]]  # Optional numeric features (if applicable)
y_category = data["Ticket priority"]  # Target variable

# Vectorize text data
vectorizer = TfidfVectorizer()
X_features_text = vectorizer.fit_transform(X_text)

# Combine text and numeric features
X_features = pd.concat([X_features_text, X_numeric], axis=1)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_features, y_category, test_size=0.2)

# Train Logistic Regression model
model_category = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_category.fit(X_train, y_train)

def predict_all(text):
    text_features = vectorizer.transform([preprocess_text(text)])
    category = model_category.predict(text_features)[0]
    sentiment = TextBlob(text).sentiment.polarity  # Sentiment analysis using TextBlob
    priority = define_priority_rules(urgency_words=["downtime", "urgent"], critical_words=["data loss"])(text)
    return category, sentiment, priority

# Test the prediction function with a new ticket
new_ticket = "My internet connection has been down for an hour. I'm very frustrated!"
category, sentiment, priority = predict_all(new_ticket)

print("Predicted category:", category)
print("Sentiment:", sentiment)  # Positive: > 0, Negative: < 0
print("Priority:", priority)
