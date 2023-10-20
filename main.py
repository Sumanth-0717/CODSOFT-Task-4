import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("advertising.csv")

# Display the first few rows of the dataset to get a quick overview
print("First few rows of the dataset:")
print(data.head())

print("Last few rows of the dataset")
print(data.tail())

print("\nSummary statistics of the dataset:")
print(data.describe())

#Data Visualisation

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot TV vs. Sales
axes[0].scatter(data['TV'], data['Sales'])
axes[0].set_xlabel('TV Advertising Budget')
axes[0].set_ylabel('Sales')
axes[0].set_title('TV vs. Sales')

# Plot Radio vs. Sales
axes[1].scatter(data['Radio'], data['Sales'])
axes[1].set_xlabel('Radio Advertising Budget')
axes[1].set_ylabel('Sales')
axes[1].set_title('Radio vs. Sales')

# Plot Newspaper vs. Sales
axes[2].scatter(data['Newspaper'], data['Sales'])
axes[2].set_xlabel('Newspaper Advertising Budget')
axes[2].set_ylabel('Sales')
axes[2].set_title('Newspaper vs. Sales')

plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a linear regression model and train it on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Create a DataFrame to store the actual and predicted sales values
predictions_df = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})

# Save the predictions to a CSV file
predictions_df.to_csv("sales_predictions.csv", index=False)

# Print the first few rows of the predictions DataFrame
print("\nFirst few rows of the predictions DataFrame:")
print(predictions_df.head())

# Evaluate the model by calculating Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
