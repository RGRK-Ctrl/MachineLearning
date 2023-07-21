
#profit.py

import mlflow
from mlflow import set_experiment
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Iris dataset
data = pd.read_csv('/Users/goutham-18258/Desktop/Jupyter notebook/50_Startups_org.csv')

encoder = ce.OneHotEncoder(cols='State', handle_unknown='return_nan', return_df=True, use_cat_names=True)
data_encoded = encoder.fit_transform(data)

data_encoded.head()

x = data_encoded.iloc[:, :-1]
y = data_encoded.iloc[:, -1]

# Step 2: Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#mlflow.set_tracking_uri("/Users/goutham-18258/PycharmProjects/Webapp/model")

# Step 3: Initialize MLflow tracking and set up the experiment
mlflow.set_experiment("profit_prediction_experiment1")

# Step 5: Start the MLflow run
with mlflow.start_run():
    # Step 7: Create and train the logistic regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Step 8: Make predictions on the test set
    y_pred = model.predict(x_test)

    fig, ax = plt.subplots()
    scatter = ax.scatter(y_pred, y_test, edgecolors=(0, 0, 1))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig('scatter.png')
    plt.close()

    # Saving plot
    mlflow.log_artifact('scatter.png', "scatter")  # Log the image as an artifact in the "scatter" directory

    # Step 9: Log evaluation metrics
    # Mean Squared Error (MSE)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("RootMeanSquaredError", rmse)

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("MeanAbsoluteError", mae)

    # R-squared (Coefficient of Determination)
    r_squared = r2_score(y_test, y_pred)
    mlflow.log_metric("RsquaredR2", r_squared)

    # Step 10: Log the trained model as an artifact
    mlflow.sklearn.log_model(model, "profit_model_2")

#app.py

import csv
import pickle
import pandas as pd
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Load the ML model from the saved file
with open("/Users/goutham-18258/PycharmProjects/Webapp/mlruns/210906563894328586/ac7a3283e12245dfb7e2e5022b7ef1d4/artifacts/profit_model_2/model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the columns used for training the model
columns_for_training = ['R&D Spend', 'Administration', 'Marketing Spend', 'State_New York','State_California', 'State_Florida']

@app.route("/", methods=["GET", "POST"])
def registration_form():
    if request.method == "POST":
        # Get user inputs from the form
        rd_spend = float(request.form["rd_spend"])
        administration = float(request.form["administration"])
        marketing_spend = float(request.form["marketing_spend"])
        state = request.form["state"]

        # Create a DataFrame with user inputs and perform one-hot encoding
        user_data = pd.DataFrame({
            "R&D Spend": [rd_spend],
            "Administration": [administration],
            "Marketing Spend": [marketing_spend],
            "State": [state]  # Include the 'State' as a string, not one-hot encoded
        })

        # Perform one-hot encoding on the 'State' column
        user_data = pd.get_dummies(user_data, columns=["State"])

        # Reorder the columns to match the model's feature order
        missing_cols = set(columns_for_training) - set(user_data.columns)
        for col in missing_cols:
            user_data[col] = 0
        user_data = user_data[columns_for_training]

        # Use the model to predict the profit
        predicted_profit = model.predict(user_data)[0]

        # Append the new entry and predicted profit to the '50_startups_org.csv' file
        with open("50_startups_org.csv", mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([rd_spend, administration, marketing_spend, state, predicted_profit])

        # Redirect to the thank-you page with the predicted profit
        return redirect(f"/thank_you?profit={predicted_profit}")

    return render_template("registration_form.html")

@app.route("/thank_you")
def thank_you_page():
    # Get the predicted profit from the query parameter
    predicted_profit = request.args.get("profit")
    return render_template("thank_you.html", profit=predicted_profit)


if __name__ == '__main__':
    app.run(port=5050, debug=True)
