from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import evaluate_model_performance

def train_models(X_train, Y_train):
    """
    Train multiple machine learning models: Logistic Regression, Decision Tree, and Random Forest.
    
    Args:
    - X_train: array-like, training data for features.
    - Y_train: array-like, training data for labels.
    
    Returns:
    - dict: A dictionary of trained models.
    """
    models = {}

    # Logistic Regression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    models['logistic_regression'] = log

    # Decision Tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)
    models['decision_tree'] = tree

    # Random Forest
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)
    models['random_forest'] = forest

    return models

def evaluate_models(models, X_test, Y_test):
    """
    Evaluate trained models and print classification reports and accuracy scores.
    
    Args:
    - models: dict, trained models.
    - X_test: array-like, test data for features.
    - Y_test: array-like, test data for labels.
    """
    for name, model in models.items():
        print(f'\nModel: {name}')
        y_pred = model.predict(X_test)
        evaluate_model_performance(Y_test, y_pred)

def predict_model(model, X_test):
    """
    Make predictions using the specified model on the test data.
    
    Args:
    - model: trained model.
    - X_test: array-like, test data for features.
    
    Returns:
    - array: Predictions made by the model.
    """
    return model.predict(X_test)
