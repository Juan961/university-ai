import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mse(predicted, real, m):
    return np.sum( (predicted - real) ** 2 ) / ( 2 * m )


def split_data(data, train_split:float=0.8, test_split:float=0.1):
    if train_split + test_split > 1.0:
        raise Exception("Invalid split data parameters")

    shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_end = int(len(shuffled) * train_split)
    test_end = train_end + int(len(shuffled) * test_split)
    
    train = shuffled.iloc[:train_end]
    test = shuffled.iloc[train_end:test_end]
    val = shuffled.iloc[test_end:]

    return train, test, val


def train(train_df, test_df, feature_col, target_col, epochs, lr=0.01):
    train_loss = []
    test_loss = []

    x = train_df[feature_col]
    y = train_df[target_col]
    m = len(y)

    w = 0.0
    b = 0.0

    for _ in range(epochs):
        predicted = w * x + b

        dw = (1/m) * np.sum((predicted - y) * x)
        db = (1/m) * np.sum(predicted - y)

        w = w - lr * dw
        b = b - lr * db

        # Train loss
        train_predicted = w * x + b
        train_loss.append(mse(train_predicted, y, m))

        # Test loss
        test_loss.append(test(test_df, w, b, feature_col, target_col))

    return train_loss, test_loss, w, b


def test(data, w, b, feature_col, target_col):
    x = data[feature_col]
    y = data[target_col]

    m = len(y)
    predicted = w * x + b

    return mse(predicted, y, m)


def main():
    feature_col = "Bedrooms"
    target_col = "Price"

    df = pd.read_csv("dataset/HousePrices.csv")

    print(df.head())

    # Create a plot of bedrooms vs price
    plt.scatter(df[feature_col], df[target_col])
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.title("Bedrooms vs Price")

    plt.figure()

    epochs = 50

    train_df, test_df, val_df = split_data(df)

    train_loss, test_loss, w, b = train(train_df, test_df, feature_col, target_col, epochs)

    plt.plot([x for x in range(epochs)], train_loss, label="Train Loss")
    plt.plot([x for x in range(epochs)], test_loss, label="Test Loss")
    plt.legend()

    plt.figure()

    mse_val = test(val_df, w, b, feature_col, target_col)

    print(f"MSE: {mse_val}")

    print(f"Learned parameters: w = {w}, b = {b}")

    # Print the line and data points
    x = np.linspace(df[feature_col].min(), df[feature_col].max(), 100)
    y = w * x + b
    plt.scatter(df[feature_col], df[target_col], label="Data Points")
    plt.plot(x, y, color="red", label="Fitted Line")
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.title("Bedrooms vs Price with Fitted Line")
    plt.legend()

    # Test an example with 4 bedrooms:
    predicted_price = w * 4 + b
    print(f"Predicted price for a house with 4 bedrooms: {predicted_price}")

if __name__ == "__main__":
    main()

    plt.show()
