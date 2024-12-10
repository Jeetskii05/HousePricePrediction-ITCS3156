from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR  
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor 
import io
import base64
from sklearn.metrics import pairwise_distances

app = Flask(__name__)

df = pd.read_csv('kc_house_data.csv')
df.dropna(subset=['price'], inplace=True)
df.drop(columns=['id', 'date'], inplace=True)
df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)

X = df.drop(columns=['price'])
y = np.log1p(df['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_model = LinearRegression().fit(X_train_scaled, y_train)
svr_model = SVR(kernel='rbf').fit(X_train_scaled, y_train)

nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)


def knn_regression(X_train, X_test, y_train, k=5):
    distances = np.linalg.norm(X_train - X_test.reshape(1,-1), axis =1)
    k_indices = np.argsort(distances)[:k]
    nearest_labels = y_train.iloc[k_indices]
    return np.mean(nearest_labels)


class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X_train, y_train):
        residuals = y_train - np.mean(y_train)
        predictions = np.full_like(y_train, np.mean(y_train), dtype=np.float64)

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X_train, residuals)
            predictions += self.learning_rate * tree.predict(X_train)
            residuals = y_train - predictions
            self.models.append(tree)

        self.final_predictions = predictions

    def predict(self, X_test):
        predictions = np.full_like(np.zeros(X_test.shape[0]), np.mean(y_train), dtype=np.float64)
        for tree in self.models:
            predictions += self.learning_rate * tree.predict(X_test)
        return predictions

gb_model = GradientBoosting(n_estimators=100, learning_rate=0.1)
gb_model.fit(X_train_scaled, y_train)

def create_actual_prices_plot():
    plt.figure(figsize=(12, 6))
    plt.hist(df['price'], bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Actual Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    actual_prices_plot = base64.b64encode(img.getvalue()).decode('utf8')
    return actual_prices_plot

def create_plots():
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, linear_model.predict(X_test_scaled), color='blue', alpha=0.5)
    plt.title("Linear Regression: Actual vs Predicted Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    linear_reg_plot = base64.b64encode(img.getvalue()).decode('utf8')

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, svr_model.predict(X_test_scaled), color='green', alpha=0.5)
    plt.title("Support Vector Regression: Actual vs Predicted Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    svr_plot = base64.b64encode(img.getvalue()).decode('utf8')

    plt.figure(figsize=(12, 6))
    knn_preds = [knn_regression(X_train_scaled, X_test_scaled[i], y_train, k=5) for i in range(X_test_scaled.shape[0])]
    plt.scatter(y_test, knn_preds, color='red', alpha=0.5)
    plt.title("Manual KNN: Actual vs Predicted Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    knn_plot = base64.b64encode(img.getvalue()).decode('utf8')

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, gb_model.predict(X_test_scaled), color='purple', alpha=0.5)
    plt.title("Gradient Boosting: Actual vs Predicted Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    gb_plot = base64.b64encode(img.getvalue()).decode('utf8')

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, nn_model.predict(X_test_scaled), color='orange', alpha=0.5)
    plt.title("Neural Network: Actual vs Predicted Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    nn_plot = base64.b64encode(img.getvalue()).decode('utf8')

    return linear_reg_plot, svr_plot, knn_plot, gb_plot, nn_plot

@app.route('/', methods=['GET', 'POST'])
def index():
    initial_data = df.head(10).to_html(classes='table table-striped', index=False)
    actual_prices_plot = create_actual_prices_plot()

    if request.method == 'POST':
        sqft_living = float(request.form['sqft_living'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        sqft_lot = float(request.form['sqft_lot'])
        floors = int(request.form['floors'])
        age = int(request.form['age'])
        waterfront = int(request.form['waterfront'])
        view = int(request.form['view'])

        new_house = pd.DataFrame({
            'sqft_living': [sqft_living],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_lot': [sqft_lot],
            'floors': [floors],
            'age': [age],
            'waterfront': [waterfront],
            'view': [view]
        })

        new_house_encoded = pd.get_dummies(new_house, drop_first=True)
        new_house_encoded = new_house_encoded.reindex(columns=X.columns, fill_value=0)
        new_house_scaled = scaler.transform(new_house_encoded)
        REALISTIC_MIN_PRICE = 10_000  
        REALISTIC_MAX_PRICE = 5_000_000  

        linear_regression_log_pred = linear_model.predict(new_house_scaled)[0]
        linear_regression_pred = np.expm1(linear_regression_log_pred)
        linear_regression_pred = max(min(linear_regression_pred, REALISTIC_MAX_PRICE), REALISTIC_MIN_PRICE)

        svr_log_pred = svr_model.predict(new_house_scaled)[0]
        svr_pred = np.expm1(svr_log_pred)
        svr_pred = max(min(svr_pred, REALISTIC_MAX_PRICE), REALISTIC_MIN_PRICE)

        knn_pred = np.expm1(knn_regression(X_train_scaled, new_house_scaled[0], y_train, k=5))
        knn_pred = max(min(knn_pred, REALISTIC_MAX_PRICE), REALISTIC_MIN_PRICE)

        gb_log_pred = gb_model.predict(new_house_scaled)[0]
        gb_pred = np.expm1(gb_log_pred)
        gb_pred = max(min(gb_pred, REALISTIC_MAX_PRICE), REALISTIC_MIN_PRICE)

        nn_log_pred = nn_model.predict(new_house_scaled)[0]
        nn_pred = np.expm1(nn_log_pred)  
        nn_pred = max(min(nn_pred, REALISTIC_MAX_PRICE), REALISTIC_MIN_PRICE)

        predictions = {
            'Manual Gradient Boosting': gb_pred,
            'Manual KNN': knn_pred,
            'Support Vector Regression': svr_pred,

        }

        linear_reg_plot, svr_plot, knn_plot, gb_plot, nn_plot = create_plots()

        return render_template('index.html', predictions=predictions,
                               linear_reg_plot=linear_reg_plot,
                               svr_plot=svr_plot,
                               knn_plot=knn_plot,
                               gb_plot=gb_plot,
                               nn_plot=nn_plot,
                               actual_prices_plot=actual_prices_plot,
                               initial_data=initial_data)

    return render_template('index.html', predictions=None,
                           actual_prices_plot=actual_prices_plot,
                           initial_data=initial_data)

if __name__ == '__main__':
    app.run(debug=True)
