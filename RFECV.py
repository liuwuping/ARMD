import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Read the data.
raw = []
with open("path") as f:
    reader = csv.reader(f)
    for row in reader:
        raw.append(row)
    raw_data = pd.DataFrame(raw[1:])
raw_features = raw_data.drop([0], axis=1)
x = raw_features.values
y = raw_data[0].values
knn = OneVsOneClassifier(KNeighborsClassifier())
svc = SVC(kernel="linear")
min_features_to_select = 1
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy',
              min_features_to_select=min_features_to_select)
rfecv.fit(x, y)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()
