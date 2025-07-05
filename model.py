import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

def main():
    print("Loading data...")
    df = pd.read_csv('master_india.csv')

    print("Encoding categorical data...")
    le = LabelEncoder()
    df['State'] = le.fit_transform(df['State'])

    print("Splitting dataset...")
    X = df.drop(['Rate of Suicides'], axis=1)
    y = df['Rate of Suicides']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    print("Saving model and encoder...")
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("✅ Model training complete and saved!")

if __name__ == "__main__":
    main()
