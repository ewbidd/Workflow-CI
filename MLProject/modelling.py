import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

df = pd.read_csv("california_housing_preprocessing.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mlflow.log_param("n_estimators", best_params['n_estimators'])
mlflow.log_param("max_depth", best_params['max_depth'])
mlflow.log_param("min_samples_split", best_params['min_samples_split'])
mlflow.log_param("cv_folds", 3)

mlflow.log_metric("mse", mse)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("r2_score", r2)

feat_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_importance.plot(kind='bar')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
mlflow.log_artifact('feature_importance.png')

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()
mlflow.log_artifact('actual_vs_predicted.png')

mlflow.sklearn.log_model(best_model, "model")

print(f"Best Params : {best_params}")
print(f"MSE         : {mse:.4f}")
print(f"RMSE        : {rmse:.4f}")
print(f"MAE         : {mae:.4f}")
print(f"R2          : {r2:.4f}")
print("CI Training selesai!")