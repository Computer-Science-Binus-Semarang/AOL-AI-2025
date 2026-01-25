import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint

df = pd.read_csv("dataset3.csv")

print("Jumlah data:", df.shape)
print("Kolom:", df.columns.tolist())
print("=" * 50)

TARGET_COL = "Workout_Type"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

FEATURE_COLS = list(X.columns)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numerical Features:", numeric_features)
print("Categorical Features:", categorical_features)
print("=" * 50)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

knn = KNeighborsClassifier()

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", knn)
    ]
)

param_dist = {
    "model__n_neighbors": randint(3, 15),
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2]  # Manhattan vs Euclidean
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,          # cukup besar untuk hasil bagus
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Mulai training model...")
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Training selesai!")
print("=" * 50)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("AKURASI MODEL:", round(accuracy * 100, 2), "%")
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

joblib.dump(best_model, "workout_model.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")

print("=" * 50)
print("Model dan feature_cols berhasil disimpan!")
print("File output:")
print("- workout_model.pkl")
print("- feature_cols.pkl")