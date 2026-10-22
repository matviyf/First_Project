from pathlib import Path
import pandas as pd

file_path = Path(__file__).parent / "apartments_10K.csv"

data = pd.read_csv(
    file_path,
    sep=";",
    encoding="cp1252"
)

print(data.head())
print(data.columns)
print(data.shape)

print(data[["square_feet", "price"]].isnull().sum())
print(data[["square_feet", "price"]].describe())
print(data["currency"].value_counts())
print(data["price_type"].value_counts())

clean_data = data[data["price_type"] == "Monthly"].copy()

clean_data = clean_data[
    (clean_data["square_feet"] >= 200) &
    (clean_data["square_feet"] <= 3000) &
    (clean_data["price"] >= 300) &
    (clean_data["price"] <= 10000)
]

print("\nOriginal dataset:", len(data))
print("Cleaned dataset:", len(clean_data))
print(clean_data[["square_feet", "price"]].describe())

X = clean_data[["square_feet"]].to_numpy()
y = clean_data["price"].to_numpy()

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 X values:", X[:5])
print("First 5 y values:", y[:5])