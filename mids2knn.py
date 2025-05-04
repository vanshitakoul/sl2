import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('Mental-Health-Twitter.csv')

# Keep only necessary columns and drop missing values
df = df[['post_text', 'label']].dropna()

# Filter binary sentiment values (0 = negative, 1 = positive)
df = df[df['label'].isin([0, 1])]

# Define custom positive and negative words
positive_words = {'happy', 'great', 'awesome', 'love', 'good', 'excellent', 'fantastic'}
negative_words = {'sad', 'bad', 'terrible', 'hate', 'awful', 'disappointed', 'poor'}

# Define stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    tokens = word_tokenize(str(text).lower())
    return ' '.join([t for t in tokens if t in positive_words or t in negative_words])

# Clean the text
df['cleaned'] = df['post_text'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)  # You can change n_neighbors for tuning
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluation metrics
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Optional: Save confusion matrix
plt.savefig('confusion_matrix.png')
plt.close()
print("✅ Confusion matrix saved as 'confusion_matrix.png'")
