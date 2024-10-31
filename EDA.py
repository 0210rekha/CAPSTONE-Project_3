# Assuming df_cleaned is already in your environment

# a) Descriptive Statistics
def descriptive_statistics(df):
    print("Descriptive Statistics:")
    print(df.describe(include='all'))
    print("\nMissing Values:\n", df.isnull().sum())

descriptive_statistics(df_cleaned)

# b) Data Visualization

sns.set(style='whitegrid')

# i) Histograms for numerical features
numerical_columns = ['Kms Driven', 'Price', 'Registration Year',
                     'Engine Displacement', 'Mileage', 'Engine',
                     'Max Power', 'Torque', 'Seating Capacity', 'Wheel Size']

# Apply winsorization to all numerical columns
for col in numerical_columns:
    df_cleaned[col] = mstats.winsorize(df_cleaned[col], limits=[0.05, 0.05])


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns):
    plt.subplot(3, 4, i + 1)
    sns.histplot(df_cleaned[col], kde=True, bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# ii) Box plots to identify outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(y=df_cleaned[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# iii) Scatter plots to examine relationships
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns):
    if col != 'Price': 
        plt.subplot(3, 4, i + 1)
        sns.scatterplot(data=df_cleaned, x=col, y='Price')
        plt.title(f'{col} vs Price')
        plt.xlabel(col)
        plt.ylabel('Price')
plt.tight_layout()
plt.show()

# iv) Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df_cleaned[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# c) Feature Selection

# Prepare features and target variable
X = df_cleaned.drop(columns=['Price'])
y = df_cleaned['Price']

# Fit a Random Forest Regressor to identify feature importance
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# Feature importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Features Importance')
plt.xlabel('Importance Score')
plt.show()