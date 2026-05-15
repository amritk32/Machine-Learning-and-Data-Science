import pandas as pd
import joblib

model = joblib.load("artifacts/model.pkl")
encoder = joblib.load("artifacts/encoder.pkl")
train_columns = joblib.load("artifacts/train_columns.pkl")

def preprocess_input(df):

    df['Stint_Fatigue'] = df['Stint'] * df['LapNumber']
    df['Pace_Efficiency'] = df['Stint'] / (df['LapTime (s)'] + 0.001)
    df['Tire_Danger_Index'] = (
        df['Cumulative_Degradation'] * df['TyreLife']
    )

    df = df.drop(
        columns=['id', 'Driver', 'RaceProgress'],
        errors='ignore'
    )

    cols_to_encode = ['Compound', 'Race']

    encoded = encoder.transform(df[cols_to_encode])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cols_to_encode),
        index=df.index
    )

    df = pd.concat(
        [df.drop(columns=cols_to_encode), encoded_df],
        axis=1
    )

    df = df.reindex(columns=train_columns, fill_value=0)

    return df


def predict(df):

    processed_df = preprocess_input(df)

    probs = model.predict_proba(processed_df)[:, 1]

    return probs
