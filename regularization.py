import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


# ==========================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.20,
    random_state=42
)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 4. DIFFERENT REGULARIZATION STRENGTHS
# ==========================================

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

results = []


# ==========================================
# 5. TRAIN RIDGE MODELS
# ==========================================

for alpha in alphas:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Performance
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)

    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)

    # Coefficients
    coefficients = model.named_steps["ridge"].coef_
    coefficient_norm = np.linalg.norm(coefficients)

    results.append([
        alpha,
        train_r2,
        val_r2,
        train_mse,
        val_mse,
        coefficient_norm
    ])


# ==========================================
# 6. CREATE RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Alpha",
        "Training_R2",
        "Validation_R2",
        "Training_MSE",
        "Validation_MSE",
        "Coefficient_Norm"
    ]
)

print("\n===== RESULTS =====")
print(results_df.to_string(index=False))


# ==========================================
# 7. FIND BEST ALPHA
# ==========================================

best_index = results_df["Validation_R2"].idxmax()

best_alpha = results_df.loc[best_index, "Alpha"]

print("\nBest Alpha:", best_alpha)


# ==========================================
# 8. TRAIN FINAL MODEL
# ==========================================

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=best_alpha))
])

final_model.fit(
    X_temp,
    y_temp
)


# ==========================================
# 9. FINAL TEST
# ==========================================

test_pred = final_model.predict(X_test)

final_test_r2 = r2_score(y_test, test_pred)
final_test_mse = mean_squared_error(y_test, test_pred)

print("\n===== FINAL TEST RESULTS =====")
print("Final Alpha:", best_alpha)
print("Final Test R²:", final_test_r2)
print("Final Test MSE:", final_test_mse)


# ==========================================
# 10. GRAPH 1 — ALPHA VS VALIDATION R²
# ==========================================

plt.figure()

plt.semilogx(
    results_df["Alpha"],
    results_df["Validation_R2"],
    marker="o"
)

plt.xlabel("Regularization Strength (Alpha)")
plt.ylabel("Validation R²")
plt.title("Effect of Regularization on Validation Performance")
plt.grid(True)

plt.show()


# ==========================================
# 11. GRAPH 2 — ALPHA VS VALIDATION MSE
# ==========================================

plt.figure()

plt.semilogx(
    results_df["Alpha"],
    results_df["Validation_MSE"],
    marker="o"
)

plt.xlabel("Regularization Strength (Alpha)")
plt.ylabel("Validation MSE")
plt.title("Effect of Regularization on Validation Error")
plt.grid(True)

plt.show()


# ==========================================
# 12. GRAPH 3 — ALPHA VS COEFFICIENT NORM
# ==========================================

plt.figure()

plt.semilogx(
    results_df["Alpha"],
    results_df["Coefficient_Norm"],
    marker="o"
)

plt.xlabel("Regularization Strength (Alpha)")
plt.ylabel("Coefficient Norm")
plt.title("Effect of Regularization on Model Complexity")
plt.grid(True)

plt.show()