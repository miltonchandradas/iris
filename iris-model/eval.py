import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression

iris = load_iris()

model_params = {
    "svm": {
        "model": svm.SVC(gamma="auto"),
        "params": {
            "C": [1, 10, 20],
            "kernel": ["rbf", "linear"]
        }
    },
    "logistic_regression": {
        "model": LogisticRegression(solver="liblinear", multi_class="auto"),
        "params": {
            "C": [1, 5, 10]
        }
    },
    "kneighbors_regressor": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
    }
}

scores = []

for key, model_type in model_params.items():
    classifier = GridSearchCV(
        model_type["model"], model_type["params"], cv=5, return_train_score=False)
    classifier.fit(iris.data, iris.target)

    scores.append({
        "model": key,
        "best_score": classifier.best_score_,
        "best_params": classifier.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
