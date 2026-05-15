import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class BaselineTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []

        return None
    
    @staticmethod 
    def train_feature_engineering(df):
        df['Stint_Fatigue'] = df['Stint'] * df['LapNumber']
        df['Pace_Efficiency'] = df['Stint'] / (df['LapTime (s)'] + 0.001)
        df['Tire_Danger_Index'] = df['Cumulative_Degradation'] * df['TyreLife']
    
        df = df.drop(columns=['id', 'Driver','RaceProgress'], errors='ignore')
        
        df = df.dropna()
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        train_temp = X_train.copy()
        train_temp[TARGET_COL] = y_train
        compound_pit_proba = (train_temp.groupby('Compound')[TARGET_COL].mean().to_dict())

        X_train['Compound_Pit_Probablity'] = X_train['Compound'].map(compound_pit_proba)
        X_train['Z'] = X_train['Compound_Pit_Probablity']*X_train['LapTime_Delta']
        X_test['Compound_Pit_Probablity'] = X_test['Compound'].map(compound_pit_proba)
        X_test['Z'] = X_test['Compound_Pit_Probablity']*X_test['LapTime_Delta']
        X_train['Is_Tyre_Old_Risk'] = (X_train['TyreLife'] > 2).astype(int)
        X_train['Critical_Pit_Alarm'] = X_train['Is_Tyre_Old_Risk'] * X_train['LapTime_Delta']
        X_test['Is_Tyre_Old_Risk'] = (X_test['TyreLife'] > 2).astype(int)
        X_test['Critical_Pit_Alarm'] = X_test['Is_Tyre_Old_Risk'] * X_test['LapTime_Delta']

        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def prepare_test_data(df):

        df['Stint_Fatigue'] = df['Stint'] * df['LapNumber']
        df['Pace_Efficiency'] = df['Stint'] / (df['LapTime (s)'] + 0.001)
        df['Tire_Danger_Index'] = df['Cumulative_Degradation'] * df['TyreLife']

        df = df.drop(columns=['id', 'Driver', 'RaceProgress'], errors='ignore')

        df = df.dropna()

        return df


    @staticmethod
    def encode_features(X_train, X_test, cols_to_encode):

        encoder = OneHotEncoder(
            sparse_output=False,
            drop='first',
            dtype=int
        )

        encoded_train = encoder.fit_transform(X_train[cols_to_encode])
        encoded_test = encoder.transform(X_test[cols_to_encode])
        train_columns = X_train.columns
        train_encoded_df = pd.DataFrame(
            encoded_train,
            columns=encoder.get_feature_names_out(cols_to_encode),
            index=X_train.index
        )

        
        test_encoded_df = pd.DataFrame(
            encoded_test,
            columns=encoder.get_feature_names_out(cols_to_encode),
            index=X_test.index
        )

        X_train = pd.concat(
            [X_train.drop(columns=cols_to_encode), train_encoded_df],
            axis=1
        )

        X_test = pd.concat(
            [X_test.drop(columns=cols_to_encode), test_encoded_df],
            axis=1
        )

        return X_train, X_test, encoder

    def _evaluate_and_print(self, model_name, model):
        print(f"\n🚀 Training {model_name}...")
        
        # Fit
        model.fit(self.X_train, self.y_train)
        
        # Predict Hard Classes (0 or 1)
        y_pred = model.predict(self.X_test)
        
        # 🚨 THE SMART PROBABILITY CHECK 🚨
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(self.X_test)
        else:
            y_pred_proba = y_pred 

        # Calculate Metrics
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        
        # Store metrics in the results list (F1 removed completely)
        self.results.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "ROC-AUC": round(auc_score, 4) 
        })
        
        # Print Metrics directly
        print(f"✅ {model_name} Results:")
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   ROC-AUC   : {auc_score:.4f}")
        print("-" * 40)
        return model
        
    def print_summary_table(self):
        if not self.results:
            print("⚠️ Bhaiwa, abhi tak kouno model train nahi hua ba!")
            return
            
        print("\n" + "="*50)
        print("🏆 FINAL BASELINE METRICS SUMMARY 🏆")
        print("="*50)
        results_df = pd.DataFrame(self.results)
        
        # 🚨 MAJOR FIX: Ab F1-Score nahi, ROC-AUC ke hisaab se sorting hogi!
        results_df = results_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
        print(results_df.to_string(index=False))
        print("="*50 + "\n")

    def get_xgb(self):
        model = XGBClassifier(
            n_estimators=300, learning_rate=0.2, max_depth=6, subsample=1.0, colsample_bytree=1.0,
            gamma=0, reg_alpha=0.01, reg_lambda=10, eval_metric='logloss', random_state=42, n_jobs=-1
        )
        pipeline = Pipeline([('model',model)])
        return self._evaluate_and_print("XGBoost", pipeline)

    def get_lightgbm(self):
        model = LGBMClassifier(
            n_estimators=290, learning_rate=0.2, num_leaves=55, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, random_state=42, n_jobs=-1, verbose=-1
        )
        return self._evaluate_and_print("LightGBM", model)

    def get_random_forest(self):
        model = RandomForestClassifier(
            n_estimators=150, criterion='gini', max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42
        )
        return self._evaluate_and_print("Random Forest", model)


# ==========================================
# Main Function
# ==========================================

if __name__ == "__main__":
    
    # 1. LOAD DATA
    df = pd.read_csv("/home/amritkg9009/Downloads/playground-series-s6e5/train.csv")
    TARGET_COL = "PitNextLap" 

    X_train, X_test, y_train, y_test = BaselineTrainer.train_feature_engineering(df)
    cols_to_encode = ['Compound', 'Race']

    X_train, X_test, encoder = BaselineTrainer.encode_features(X_train,X_test,cols_to_encode)
 
    obj = BaselineTrainer(X_train, X_test, y_train, y_test)

    o1 = obj.get_xgb()
    o4 = obj.get_random_forest()
    o5 = obj.get_lightgbm()
    

    print("\n🚀 Training Master Voting Classifier...")
    er = VotingClassifier(
        estimators=[('xgb', o1), ('rf', o4), ('lgbm', o5)], 
        voting='soft',
    )

    er.fit(X_train, y_train)
    y_pred_er = er.predict(X_test)
    y_pred_proba_er = er.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_pred_proba_er)
    acc = accuracy_score(y_test, y_pred_er)
    prec = precision_score(y_test, y_pred_er, zero_division=0)
    rec = recall_score(y_test, y_pred_er, zero_division=0)
# 
    print(f"👑 Voting Classifier Results:")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   ROC-AUC   : {auc_score:.4f}")
    print("-" * 40)
    
    obj.print_summary_table()

    joblib.dump(er, "artifacts/model.pkl")
    joblib.dump(encoder, "artifacts/encoder.pkl")
    joblib.dump(X_train.columns.tolist(), "artifacts/train_columns.pkl")
    print("✅ Model artifacts saved successfully!")


# ===============================
# Kaggle Test Submission Pipeline
# ===============================

TEST_PATH = "/home/amritkg9009/Downloads/playground-series-s6e5/test.csv"
df_final = pd.read_csv(TEST_PATH)
df_final = pd.read_csv(TEST_PATH)
test_ids = df_final['id']
df_final = BaselineTrainer.prepare_test_data(df_final)

# IMPORTANT: reuse encoder from TRAIN
cols_to_encode = ['Compound', 'Race']
encoded_test = encoder.transform(df_final[cols_to_encode])
train_cols = X_train.columns
encoded_test_df = pd.DataFrame(
    encoded_test,
    columns=encoder.get_feature_names_out(cols_to_encode),
    index=df_final.index
)
df_final = pd.concat(
    [df_final.drop(columns=cols_to_encode), encoded_test_df],
    axis=1
)

df_final = df_final.reindex(columns=train_cols, fill_value=0)

# FINAL PREDICTION USING TRAINED MODEL (er already trained)
final_probs = er.predict_proba(df_final)[:, 1]

submission = pd.DataFrame({
    "id": test_ids,
    "PitNextLap": final_probs
})

submission.to_csv("submission.csv", index=False)

print("✅ Submission file created successfully!")

