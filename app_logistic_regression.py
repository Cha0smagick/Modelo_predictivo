# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Configure the data URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Define column names
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

# Read data from the URL
data = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Label encode the target column (income)
data["income"] = data["income"].apply(lambda x: 0 if x == "<=50K" else 1)

# Separate data into features (X) and labels (y)
X = data.drop("income", axis=1)
y = data["income"]

# Convert categorical variables to numerical
le = LabelEncoder()
categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
for column in categorical_columns:
    data[column] = le.fit_transform(data[column])

# Separate data into features (X) and labels (y)
X = data.drop("income", axis=1)
y = data["income"]

# Split data into training and testing sets without oversampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tune Logistic Regression parameters using GridSearchCV
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters:", grid_search.best_params_)

# Create and train the model with the best parameters
best_model = LogisticRegression(**grid_search.best_params_)
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate model accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("Classification report:\n", classification_report(y_test, y_pred))

# Check class ratio in training and test sets
print("Class ratio in training set:")
print(y_train.value_counts(normalize=True))

print("\nClass ratio in test set:")
print(y_test.value_counts(normalize=True))

# Analyze the dataset
print("Dataset information:")
print(data.info())

print("\nStatistical summary of the dataset:")
print(data.describe())

# Review model predictions on the test set
print("Model predictions on the test set:")
print(pd.DataFrame({"Real": y_test, "Prediction": y_pred}))
