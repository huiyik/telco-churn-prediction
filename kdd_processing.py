import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1. DATA SELECTION
print("Loading data...")
df = pd.read_csv('Telco-Customer-Churn.csv')

# 2. DATA PREPROCESSING & TRANSFORMATION
print("Cleaning data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].mean()}, inplace=True)

# 3. VARIABLE SELECTION — 8 Important Predictors Only
selected_features = [
    'Contract',
    'tenure',
    'MonthlyCharges',
    'InternetService',
    'OnlineSecurity',
    'TechSupport',
    'PaymentMethod',
    'PaperlessBilling'
]

X = df[selected_features]
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert text categories into numbers
X = pd.get_dummies(X, drop_first=True)

# Save column names for dashboard
joblib.dump(list(X.columns), 'model_columns.pkl')
print(f"Selected features after encoding: {list(X.columns)}")

# 4. DATA SPLITTING (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 5. DATA MINING - MODEL TRAINING
print("\nTraining models...")

# Decision Tree Variations
dt_default = DecisionTreeClassifier(random_state=42)
dt_depth3  = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_depth5  = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Other Models
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=5000)

# Train all
dt_default.fit(X_train, y_train)
dt_depth3.fit(X_train, y_train)
dt_depth5.fit(X_train, y_train)
dt_entropy.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# 6. EVALUATION
print("\n--- MODEL EVALUATION RESULTS ---\n")

models = {
    'Decision Tree (Default)'    : dt_default,
    'Decision Tree (Max Depth 3)': dt_depth3,
    'Decision Tree (Max Depth 5)': dt_depth5,
    'Decision Tree (Entropy)'    : dt_entropy,
    'Random Forest'              : rf,
    'Logistic Regression'        : lr,
}

print(f"{'Model':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 80)

for name, model in models.items():
    predictions = model.predict(X_test)
    acc  = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec  = recall_score(y_test, predictions)
    f1   = f1_score(y_test, predictions)
    print(f"{name:<35} {acc:>10.2f} {prec:>10.2f} {rec:>10.2f} {f1:>10.2f}")

print("\n--- CONFUSION MATRICES ---\n")
for name, model in models.items():
    predictions = model.predict(X_test)
    print(f"{name}:")
    print(confusion_matrix(y_test, predictions))
    print()

# 7. SAVE THE BEST MODEL (Logistic Regression)
joblib.dump(lr, 'churn_model.pkl')
print("Model saved as churn_model.pkl!")
print("Column structure saved as model_columns.pkl!")
