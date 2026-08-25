import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = np.arange(10).reshape(-1,1)
y = np.array([0,1,0,0,1,1,1,1,1,1])

model = LogisticRegression(solver = 'liblinear', C = 10.0, random_state = 0)
model.fit(x,y)

ppred = model.predict_proba(x)
ypred = model.predict(x)
score_ = model.score(x,y)
conf_matrix = confusion_matrix(y, ypred)
report = classification_report(y, ypred)

print("x:", x, sep = "\n")
print("y:", y, sep = "\n", end = "\n\n")
print("Intercept:", model.intercept_)
print("Coefficiant:", model.coef_, end = "\n\n")
print("y prediction:", ypred, end = "\n\n")