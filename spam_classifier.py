import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print("First 5 rows:")
print(df.head())
print("\nLabel distribution:")
print(df['label'].value_counts())

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix - Spam Detection")
plt.colorbar()
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.yticks([0, 1], ['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\n--- Testing custom messages ---")
test_messages = [
    "Congratulations! You won a free iPhone, click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?"
]
test_vec = vectorizer.transform(test_messages)
test_predictions = model.predict(test_vec)
for msg, pred in zip(test_messages, test_predictions):
    label = "SPAM" if pred == 1 else "HAM (Not Spam)"
    print(f"Message: {msg}\nPrediction: {label}\n")

print("Done! Confusion matrix saved as 'confusion_matrix.png'")