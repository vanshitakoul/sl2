import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

# Load Data
train_url = "https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv"
test_url = "https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/test.csv"

train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

# Optional: Shuffle train data and view a few rows
print("🔀 Sample shuffled training data:")
print(train_data.sample(frac=1).head(5))

# TF-IDF Vectorization
print("🔧 Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
X_train = vectorizer.fit_transform(train_data['Content'])
X_test = vectorizer.transform(test_data['Content'])
y_train = train_data['Label']
y_test = test_data['Label']

# Train SVM classifier
print("🚀 Training SVM classifier...")
classifier = svm.SVC(kernel='linear')

start_train = time.time()
classifier.fit(X_train, y_train)
end_train = time.time()

# Predict
start_pred = time.time()
y_pred = classifier.predict(X_test)
end_pred = time.time()

# Performance Timing
print(f"\n⏱ Training time: {end_train - start_train:.2f} seconds")
print(f"⏱ Prediction time: {end_pred - start_pred:.2f} seconds")

# Evaluation Report
report = classification_report(y_test, y_pred, output_dict=True)
print("\n📊 Classification Report:")
print("✅ Positive:", report.get('pos', 'N/A'))
print("❌ Negative:", report.get('neg', 'N/A'))
