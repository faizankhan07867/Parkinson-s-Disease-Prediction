import pandas as pd

from config import *


class ParkinsonDataset:

    def __init__(self):

        self.dataset_path = DATASET_PATH

    # ======================================
    # Load Dataset
    # ======================================

    def load(self):

        df = pd.read_csv(

            self.dataset_path

        )

        print("Dataset Loaded Successfully")

        print(f"Total Records : {len(df)}")

        return df

    # ======================================
    # Remove Duplicates
    # ======================================

    def clean(self, df):

        df = df.drop_duplicates()

        df = df.reset_index(

            drop=True

        )

        return df

    # ======================================
    # Handle Missing Values
    # ======================================

    def handle_missing(self, df):

        numeric_columns = df.select_dtypes(

            include="number"

        ).columns

        for column in numeric_columns:

            df[column].fillna(

                df[column].median(),

                inplace=True

            )

        return df

    # ======================================
    # Validate Dataset
    # ======================================

    def validate(self, df):

        if TARGET_COLUMN not in df.columns:

            raise ValueError(

                f"{TARGET_COLUMN} column not found."

            )

        print(

            "Dataset Validation Successful"

        )

        return df
    
    # ======================================
    # Class Distribution
    # ======================================

    def class_distribution(self, df):

        print()

        print("=" * 60)

        print("Class Distribution")

        print("=" * 60)

        print(

            df[TARGET_COLUMN]

            .value_counts()

        )

        print()

        print(

            df[TARGET_COLUMN]

            .value_counts(

                normalize=True

            ) * 100

        )
        
    # ======================================
    # Statistical Summary
    # ======================================

    def statistics(self, df):

        print()

        print("=" * 60)

        print("Statistical Summary")

        print("=" * 60)

        print(

            df.describe()

        )
        
    # ======================================
    # Correlation Matrix
    # ======================================

    def correlation(self, df):

        print()

        print("=" * 60)

        print("Correlation Matrix")

        print("=" * 60)

        correlation = df.drop(

            columns=DROP_COLUMNS,

            errors="ignore"

        ).corr()

        print(

            correlation

        )

        return correlation
    
    # ======================================
    # Dataset Summary
    # ======================================

    def summary(self, df):

        print("=" * 60)

        print(df.head())

        print()

        print(df.info())

        print()

        print("Missing Values")

        print(

            df.isnull().sum()

        )

        self.class_distribution(

            df

        )

        self.statistics(

            df

        )

        self.correlation(

            df
        )

        print("=" * 60)
        
    # ======================================
    # Complete Pipeline
    # ======================================

    def prepare(self):

        df = self.load()

        df = self.clean(df)

        df = self.handle_missing(df)

        df = self.validate(df)

        self.summary(df)

        return df


# ======================================
# Test
# ======================================

if __name__ == "__main__":

    dataset = ParkinsonDataset()

    dataframe = dataset.prepare()

    print()

    print(dataframe.head())