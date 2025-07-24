from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


def load_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    return train_df, test_df
