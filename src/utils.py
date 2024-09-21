from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy based on the true labels and predictions.
    
    Args:
    - y_true: array-like, true labels.
    - y_pred: array-like, predicted labels.
    
    Returns:
    - float: The accuracy score.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract True Positives, True Negatives, False Positives, False Negatives
    tp = cm[0][0]
    tn = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy

def evaluate_model_performance(y_true, y_pred):
    """
    Evaluate the model performance by printing classification report and accuracy.
    
    Args:
    - y_true: array-like, true labels.
    - y_pred: array-like, predicted labels.
    
    Returns:
    - None
    """
    accuracy = accuracy_score(y_true, y_pred)
    print('Classification Report:')
    print(classification_report(y_true, y_pred))
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def evaluate_all_models(models, X_test, Y_test):
    """
    Evaluate all trained models and print the performance.
    
    Args:
    - models: dict, dictionary of trained models.
    - X_test: array-like, test data.
    - Y_test: array-like, test labels.
    
    Returns:
    - None
    """
    for name, model in models.items():
        print(f'\nEvaluating Model: {name}')
        y_pred = model.predict(X_test)
        evaluate_model_performance(Y_test, y_pred)
