import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """
    Preprocess the data by removing null values, encoding categorical features,
    and splitting into training and test sets.
    
    Args:
    - df: pandas DataFrame, the data to preprocess.
    
    Returns:
    - X_train, X_test, Y_train, Y_test: Processed and split data for model training.
    """
    # Drop null columns
    df = df.dropna(axis=1)

    # Encode categorical 'diagnosis' column
    df['diagnosis'] = df['diagnosis'].astype('category').cat.codes

    # Feature Scaling
    X = df.iloc[:, 2:31].values
    Y = df.iloc[:, 1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test
