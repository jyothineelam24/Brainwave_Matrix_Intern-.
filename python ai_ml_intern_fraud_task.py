import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path='creditcard.csv'):
    return pd.read_csv(file_path)


def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    return rf, lr


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    print(f"\n=== Evaluation for {name} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))


def save_model(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def main():
    print("📥 Loading dataset...")
    df = load_data()
    print("🔄 Preprocessing...")
    X, y, scaler = preprocess_data(df)
    print("⚖️ Balancing data with SMOTE...")
    X_resampled, y_resampled = balance_data(X, y)
    print("🧪 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    print("🤖 Training models...")
    rf, lr = train_models(X_train, y_train)
    print("📊 Evaluating models...")
    evaluate_model(rf, X_test, y_test, 'Random Forest')
    evaluate_model(lr, X_test, y_test, 'Logistic Regression')
    print("💾 Saving model and scaler...")
    save_model(rf, scaler)
    print("✅ Task Completed. Artifacts saved: model.pkl, scaler.pkl")


if __name__ == '__main__':
    main()
