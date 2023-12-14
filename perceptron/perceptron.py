from template import template
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# in this example we train on breast cancer datasheet

# Load breast cancer dataset
test_dataset = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    test_dataset['data'], test_dataset['target'],test_size=0.7, random_state=0)

tool = template(X_test.shape[1],.0001)
tool.train(X_train,y_train)
print(tool.evaluate(X_test,y_test))