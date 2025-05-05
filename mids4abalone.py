import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
columns = ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']
df = pd.read_csv(url, names=columns)


df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})

df['Age'] = df['Rings'] + 1.5


# Task A: Classification (Young (<=10 rings) vs Old (>10 rings))

df['AgeGroup'] = df['Rings'].apply(lambda x: 1 if x > 10 else 0)  # 1 = old, 0 = young
X = df.drop(['Rings', 'Age', 'AgeGroup'], axis=1)
y_class = df['AgeGroup']

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
print("✅ Classification Task")
print("Classification Accuracy:", accuracy_score(y_test, y_pred_class))

# ----------------------
# Task B: Regression (Predict number of Rings and Age)
# ----------------------
y_rings = df['Rings']
X_train, X_test, y_train, y_test = train_test_split(X, y_rings, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_rings = reg.predict(X_test)

# Predict age from rings
y_pred_age = y_pred_rings + 1.5

print("\n✅ Regression Task")
print("Mean Squared Error (Rings Prediction):", mean_squared_error(y_test, y_pred_rings))
print("Sample Predicted Rings:", y_pred_rings[:5])
print("Sample Predicted Age:", y_pred_age[:5])
