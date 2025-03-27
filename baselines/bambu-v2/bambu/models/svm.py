from flaml.model import SKLearnEstimator
from flaml import tune
from sklearn.svm import SVC, SVR
import numpy as np
import shutil
import time

class SvmEstimator(SKLearnEstimator):
    """The class for tuning a Support Vector Machine (Classifier or Regressor)"""

    @classmethod
    def search_space(cls, data_size, **params):
        task = params.get("task", "classification")  # Default to classification

        # Common hyperparameters for both SVC and SVR
        search_space = {
            "C": {
                "domain": tune.uniform(lower=1e-3, upper=1e3),
                "init_value": 1e-3
            },
            "kernel": {
                "domain": tune.choice(["linear", "poly", "rbf", "sigmoid"]),
                "init_value": "rbf"
            },
            "degree": {
                "domain": tune.randint(lower=2, upper=4),
                "init_value": 2
            },
            "gamma": {
                "domain": tune.choice(["scale"]),
                "init_value": "scale"
            },
            "coef0": {
                "domain": tune.uniform(lower=1e-5, upper=1e3),
                "init_value": 1e-5
            },
            "tol": {
                "domain": tune.uniform(lower=1e-5, upper=1e5),
                "init_value": 1e-5
            },
            "shrinking": {
                "domain": tune.choice([False, True]),
                "init_value": False
            },
            "cache_size": {
                "domain": tune.uniform(lower=1e-1, upper=5e2),
                "init_value": 1e-1                
            },
            "max_iter": {
                "domain": tune.randint(lower=10, upper=10000),
                "init_value": -1               
            }
        }

        # Add task-specific parameters
        if task == "classification":
            search_space["break_ties"] = {
                "domain": tune.choice([False, True]),
                "init_value": False                  
            }
            search_space["decision_function_shape"] = {
                "domain": tune.choice(["ovo", "ovr"]),
                "init_value": "ovr"
            }
        elif task == "regression":
            search_space["epsilon"] = {
                "domain": tune.uniform(lower=1e-5, upper=1e-1),
                "init_value": 1e-5
            }
        else:
            raise ValueError(f"Invalid task type: {task}. Choose 'classification' or 'regression'.")

        cls._hyperparameters = search_space.keys()
        return search_space

    
    @classmethod
    def size(cls, config):
        return 1.0

    def config2params(self, config: dict) -> dict:
        params = config.copy()
        return params

    def __init__(
        self,
        task="classification",
        **config,
    ):
        super().__init__(task, **config)

    def fit(self, X_train, y_train, budget=None, **kwargs):
        hyperparameters = self.params.copy()
        for param in self.params:
            if param not in self._hyperparameters:
                del hyperparameters[param]
        if self._task == "classification":
            svm = SVC(**hyperparameters, probability=True)
        else:
            svm = SVR(**hyperparameters)
        start_time = time.time()
        deadline = start_time + budget if budget else np.inf
        svm.fit(X_train, y_train)        
        train_time = time.time() - start_time
        self._model = svm
        return train_time

    def predict(self, X_test):
        return super().predict(X_test)

    def predict_proba(self, X_test):
        return super().predict_proba(X_test)


