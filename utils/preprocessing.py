from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


def preprocess(df: pd.DataFrame, scaler=None, encoders=None, is_train=True):
    df = df.copy()

    # Binary classification: 'normal' vs. 'attack'
    df['class'] = df['class'].apply(
        lambda x: 'normal' if x == 'normal' else 'attack')

    # Drop NA
    df.dropna(inplace=True)

    # Extract features and classs
    X = df.drop('class', axis=1)
    y = df['class']

    if is_train:
        # Label encode all categorical columns
        encoders = {}
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # Use previously fitted encoders and scaler
        for col, le in encoders.items():
            if col in X.columns:
                X[col] = le.transform(X[col])
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, encoders
