from data_loader import load_data
from preprocessing import preprocess_data
from eda import perform_eda
from models import train_models, evaluate_models, predict_model


data_file_path = 'data.csv' #input your path for the data to be used.

def main():
    # Load data
    df = load_data(data_file_path)

    # Preprocessing
    X_train, X_test, Y_train, Y_test = preprocess_data(df)
    # Train models
    models = train_models(X_train, Y_train)
    # Evaluate models
    evaluate_models(models, X_test, Y_test)
    # Make predictions with Random Forest model
    predictions = predict_model(models['random_forest'], X_test)

    # Print predictions
    print('Our model prediction:', predictions)

if __name__ == "__main__":
    main()
