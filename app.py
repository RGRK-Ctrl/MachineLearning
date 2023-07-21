
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
        #return redirect(f"/thank_you?profit={predicted_profit}")
        return redirect(
            f"/thank_you?profit={predicted_profit}&rd_spend={rd_spend}&administration={administration}&marketing_spend={marketing_spend}&state={state}")

    return render_template("registration_form.html")

@app.route("/thank_you")
def thank_you_page():
    # Get the predicted profit from the query parameter
    predicted_profit = request.args.get("profit")

    # Get the input values from the request parameters
    rd_spend = request.args.get("rd_spend")
    administration = request.args.get("administration")
    marketing_spend = request.args.get("marketing_spend")
    state = request.args.get("state")

    return render_template("thank_you.html", rd_spend=rd_spend, administration=administration,
                           marketing_spend=marketing_spend, state=state,profit=predicted_profit)

if __name__ == '__main__':
    app.run(port=5050, debug=True)
