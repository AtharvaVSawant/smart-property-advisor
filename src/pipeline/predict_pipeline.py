import sys
import os
import pandas as pd
import pickle

from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        try:
            # ✅ Get absolute base directory
            base_dir = os.path.dirname(os.path.abspath(__file__))

            # ✅ Build correct paths (deployment safe)
            self.model_path = os.path.abspath(
                os.path.join(base_dir, "..", "..", "artifacts", "model.pkl")
            )

            self.preprocessor_path = os.path.abspath(
                os.path.join(base_dir, "..", "..", "artifacts", "preprocessor.pkl")
            )

            # ✅ Debug logs (very helpful in deployment)
            print(f"Loading model from: {self.model_path}")
            print(f"Loading preprocessor from: {self.preprocessor_path}")

            # ✅ Load model
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            # ✅ Load preprocessor
            with open(self.preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            # ✅ Transform input
            data_scaled = self.preprocessor.transform(features)

            # ✅ Predict
            preds = self.model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        CRIM: float,
        ZN: float,
        INDUS: float,
        NOX: float,
        RM: float,
        AGE: float,
        DIS: float,
        RAD: float,
        TAX: float,
        PTRATIO: float,
        B: float,
        LSTAT: float,
        CHAS: int
    ):
        self.CRIM = CRIM
        self.ZN = ZN
        self.INDUS = INDUS
        self.NOX = NOX
        self.RM = RM
        self.AGE = AGE
        self.DIS = DIS
        self.RAD = RAD
        self.TAX = TAX
        self.PTRATIO = PTRATIO
        self.B = B
        self.LSTAT = LSTAT
        self.CHAS = CHAS

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "CRIM": [self.CRIM],
                "ZN": [self.ZN],
                "INDUS": [self.INDUS],
                "NOX": [self.NOX],
                "RM": [self.RM],
                "AGE": [self.AGE],
                "DIS": [self.DIS],
                "RAD": [self.RAD],
                "TAX": [self.TAX],
                "PTRATIO": [self.PTRATIO],
                "B": [self.B],
                "LSTAT": [self.LSTAT],
                "CHAS": [self.CHAS],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)