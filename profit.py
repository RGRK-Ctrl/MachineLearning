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