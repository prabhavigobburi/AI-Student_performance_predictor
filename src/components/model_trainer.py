from sklearn.ensemble import RandomForestRegressor
from src.utils import save_object

class ModelTrainer:

    def initiate_model_trainer(self,X_train,y_train):

        model=RandomForestRegressor()

        model.fit(X_train,y_train)

        save_object(
            "artifacts/model.pkl",
            model
        )