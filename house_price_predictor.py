import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("house_data.csv")

# Input features and target
X = df[["area_sqft", "bedrooms"]]
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
input_data = pd.DataFrame(
    [[area, bedrooms]],
    columns=["area_sqft", "bedrooms"]
)

prediction = model.predict(input_data)

print(f"Predicted house price: ₹{prediction[0]:,.2f}")
