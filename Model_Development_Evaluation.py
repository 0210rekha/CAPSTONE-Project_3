import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Assuming `df_cleaned` is your cleaned dataset
# 1. Separate Features and Target
X = df_cleaned.drop('Price', axis=1)  # Features
y = df_cleaned['Price']  # Target variable

# 2. Train-Test Split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# 3. Scaling: Apply scaling to all numerical features using MinMaxScaler
scaler = StandardScaler()

# 4. Define Pipelines for Models (with Scaling for Linear Regression and PCA for Linear Regression)
models = {
    'Decision Tree': Pipeline([('scaler', scaler), ('model', DecisionTreeRegressor(random_state=42))]),
    'Random Forest': Pipeline([('scaler', scaler), ('model', RandomForestRegressor(random_state=42))]),
    'Gradient Boosting': Pipeline([('scaler', scaler), ('model', GradientBoostingRegressor(random_state=42))]),
    'Lasso Regression': Pipeline([('scaler', scaler), ('model', Lasso(alpha=0.1))]),
    'Ridge Regression': Pipeline([('scaler', scaler), ('model', Ridge(alpha=0.1))]),
    'Linear Regression': Pipeline([('scaler', scaler), ('pca', PCA(n_components=0.95, random_state=42)), ('model', LinearRegression())])
}

# 5. Model Training and Cross-Validation
print("\n==== Training Model Performance (Cross-Validation Scores) ====")
for name, pipeline in models.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"{name}: Mean R2 = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")

# 6. Train Final Models and Evaluate on Test Set
results = {}

# Train all models and evaluate performance on the test set
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

# 7. Output Test Set Performance
print("\n==== Test Set Performance ====")
for name, metrics in results.items():
    print(f"{name} -> MAE: {metrics['MAE']:.2f}, MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.4f}")

# 8. Hyperparameter Tuning for Random Forest (Optional)
param_dist = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}

random_search = RandomizedSearchCV(
    models['Random Forest'],  # Use the Random Forest pipeline
    param_distributions=param_dist,
    n_iter=10,  # Number of combinations to try
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
print("\nBest Parameters from Random Search:")
print(random_search.best_params_)

# 9. Evaluate the Best Model on the Test Set

# After hyperparameter tuning
best_model = random_search.best_estimator_

# Get the name of the best model
best_model_name = type(best_model.named_steps['model']).__name__

# Make predictions with the best model
y_pred_best = best_model.predict(X_test)

# Calculate performance metrics
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

# Print the results including the best model name
print(f"\nBest Model: {best_model_name} -> MAE: {mae_best:.2f}, MSE: {mse_best:.2f}, R²: {r2_best:.4f}")
