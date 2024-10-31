import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Separate Features and Target
X = df_cleaned.drop('Price', axis=1)
y = df_cleaned['Price']

# 2. Check for Categorical Features and Apply Encoding
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(include=[np.number]).columns),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ]
)

# 3. Train-Test Split (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# 4. Define Pipelines
lin_reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('model', LinearRegression())
])

# Pipelines for other models without PCA
models = {
    'Decision Tree': Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=42))]),
    'Random Forest': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
    'Gradient Boosting': Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingRegressor(random_state=42))]),
    'Lasso Regression': Pipeline([('preprocessor', preprocessor), ('model', Lasso(alpha=0.1))]),
    'Ridge Regression': Pipeline([('preprocessor', preprocessor), ('model', Ridge(alpha=0.1))])
}

# 5. Model Training and Cross-Validation
print("\n==== Model Performance (Cross-Validation Scores) ====")
cv_scores = cross_val_score(lin_reg_pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"Linear Regression: Mean R2 = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")

for name, pipeline in models.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"{name}: Mean R2 = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")

# 6. Train Final Models and Evaluate on Test Set
results = {}

# Linear Regression
lin_reg_pipeline.fit(X_train, y_train)
y_pred_lin = lin_reg_pipeline.predict(X_test)
results['Linear Regression'] = {
    'MAE': mean_absolute_error(y_test, y_pred_lin),
    'MSE': mean_squared_error(y_test, y_pred_lin),
    'R2': r2_score(y_test, y_pred_lin)
}

# Other Models
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

# 8. Hyperparameter Tuning for Random Forest
param_dist = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}

random_search = RandomizedSearchCV(
    models['Random Forest'],  # Use the Random Forest pipeline
    param_distributions=param_dist,
    n_iter=10, 
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

best_model = random_search.best_estimator_
best_model_name = type(best_model.named_steps['model']).__name__
y_pred_best = best_model.predict(X_test)

# Calculate performance metrics
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

# Print the results including the best model name
print(f"\nBest Model: {best_model_name} -> MAE: {mae_best:.2f}, MSE: {mse_best:.2f}, R²: {r2_best:.4f}")

# Save the model and the pipeline
joblib.dump(best_model, 'best_used_car_price_model.joblib')
joblib.dump(one_hot_encode, 'one_hot_encode.joblib')  # Save the preprocessor pipeline
print("Model saved as 'best_used_car_price_model.joblib'")
print("Preprocessor saved as 'one_hot_encode.joblib'")

joblib.dump(scaler, 'scaler.joblib')
print("Preprocessor saved as 'scaler.joblib'")