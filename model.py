import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# --- 1. Load the Data ---
# Load the dataset from the data subfolder
df = pd.read_csv("C:\\Users\\SAKSHI\\OneDrive\\Desktop\\heart-disease-prediction\\data\\heart.csv.csv")

# --- 2. Data Preprocessing ---
# Separate features (X) and target (y)
# We drop the 'target' column to create our feature set X
X = df.drop('target', axis=1) 
# The 'target' column is what we want to predict
y = df['target']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
# StandardScaler standardizes features by removing the mean and scaling to unit variance
# This is important for algorithms like Logistic Regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- 3. Model Training ---
# Initialize and train the Logistic Regression model
# We use a random_state for reproducibility
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


# --- 4. Model Evaluation ---
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print the classification report for more detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --- 5. Save the Model and Scaler ---
# We use pickle to save our trained model and scaler to disk
# The 'wb' means we are writing in binary mode
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Model and Scaler have been saved successfully!")