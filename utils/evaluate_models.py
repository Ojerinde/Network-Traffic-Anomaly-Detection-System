from sklearn.metrics import classification_report
import pandas as pd


def evaluate_model(model, X_test, y_test, is_unsupervised=False):
    if is_unsupervised:
        print("Unsupervised Evaluation")
        print(pd.Series(y_test).value_counts())
    else:
        preds = model.predict(X_test)
        print("Supervised Evaluation:")
        print(classification_report(y_test, preds))
