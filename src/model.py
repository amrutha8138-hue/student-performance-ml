import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("../data/student_data.csv")

# LINEAR REGRESSION (Marks Prediction)
X = data[['study_hours', 'attendance', 'previous_marks']]
y_marks = data['final_marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_marks, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

print("Predicted Marks:", linear_model.predict(X_test))

# LOGISTIC REGRESSION (Pass / Fail)
y_pass = data['pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_pass, test_size=0.2, random_state=42
)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

predictions = log_model.predict(X_test)
print("Pass/Fail Accuracy:", accuracy_score(y_test, predictions))
