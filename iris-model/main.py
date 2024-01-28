import pandas as pd
from sklearn.datasets import load_iris
from sklearn import svm

MODEL_PATH = '/app/model/model.pkl'

iris = load_iris()

model = svm.SVC(C=1, kernel="rbf")
model.fit(iris.data, iris.target)

score = model.score(iris.data, iris.target)
print(f"Test Data Score: {score}")

# Save model
import pickle
pickle.dump(model, open(MODEL_PATH, 'wb'))