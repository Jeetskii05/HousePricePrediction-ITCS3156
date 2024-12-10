House Price Prediction

This project utilizes machine learning techniques to predict house prices based on various features.

Features
    •    Bedrooms: Number of bedrooms
    •    Bathrooms: Number of bathrooms
    •    Size: Square footage of the house
    •    Zip Code: Location identifier
    •    Price: Target variable representing the house price

Dataset

The dataset includes the aforementioned features and is located in the kc_house_data.csv file.

Requirements
    •    Python 3.x
    •    pandas
    •    scikit-learn
    •    Flask

Install the required packages using:

pip install pandas scikit-learn Flask

Usage
    1.    Data Preprocessing: Clean and preprocess the data using pandas.
    2.    Model Training: Train machine learning models (e.g., Linear Regression) using scikit-learn.
    3.    Prediction: Use the trained model to predict house prices.
    4.    Web Interface: Run app.py to start a Flask web application for user interaction.

Files
    •    kc_house_data.csv: Dataset containing house features and prices.
    •    app.py: Flask application for the web interface.
    •    templates/: HTML templates for the web interface.
    •    static/: Static files (e.g., CSS) for the web interface.
