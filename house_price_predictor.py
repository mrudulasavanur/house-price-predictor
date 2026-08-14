import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample house price dataset
data = {
    "area": [1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "bedrooms": [2, 2, 3, 3, 4, 4, 5],
    "price": [40, 48, 60, 72, 85, 95, 110]
}

df = pd.DataFrame(data)

# Input features and target
X = df[["area", "bedrooms"]]
y = df["price"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get user input
area = float(input("Enter house area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))

# Predict house price
prediction = model.predict([[area, bedrooms]])

print(f"Predicted house price: ₹{prediction[0]:.2f} lakhs")
