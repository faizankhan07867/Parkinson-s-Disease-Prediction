import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import *


class DataPreprocessor:

    def __init__(self):

        self.scaler = StandardScaler()

    # ======================================
    # Drop Unnecessary Columns
    # ======================================

    def remove_columns(self, df):

        df = df.drop(

            columns=DROP_COLUMNS,

            errors="ignore"

        )

        return df

    # ======================================
    # Feature / Target Split
    # ======================================

    def split_features_target(self, df):

        X = df.drop(

            columns=[TARGET_COLUMN]

        )

        y = df[TARGET_COLUMN]

        return X, y

    # ======================================
    # Train Test Split
    # ======================================

    def train_test(self, X, y):

        return train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y

        )
    # ======================================
    # Feature Scaling
    # ======================================

    def scale(

        self,

        X_train,

        X_test

    ):

        X_train = self.scaler.fit_transform(

            X_train

        )

        X_test = self.scaler.transform(

            X_test

        )

        return X_train, X_test
    # ======================================
    # Save Scaler
    # ======================================

    def save(self):

        joblib.dump(

            self.scaler,

            SCALER_PATH

        )
        
    # ======================================
    # Complete Pipeline
    # ======================================

    def process(self, df):

        df = self.remove_columns(df)

        X, y = self.split_features_target(df)

        X_train, X_test, y_train, y_test = self.train_test(

            X,

            y

        )

        X_train, X_test = self.scale(

            X_train,

            X_test

        )

        self.save()

        return (

            X_train,

            X_test,

            y_train,

            y_test,

            X.columns.tolist()

        )
        
# ======================================
# Test
# ======================================

if __name__ == "__main__":

    from utils.dataset import ParkinsonDataset

    dataset = ParkinsonDataset()

    dataframe = dataset.prepare()

    processor = DataPreprocessor()

    X_train, X_test, y_train, y_test, features = processor.process(

        dataframe

    )

    print()

    print("=" * 60)

    print("Training Shape :", X_train.shape)

    print("Testing Shape  :", X_test.shape)

    print("Total Features :", len(features))

    print()

    print("Feature Names")

    print(features)

    print("=" * 60)