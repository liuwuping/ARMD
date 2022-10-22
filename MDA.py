import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Read the data.
data = pd.read_csv(r'path', encoding='gbk')
y = data.iloc[:, 0:1].values
x = data.drop('Class', axis=1)
names = data.iloc[0:0, 1:]

seed = 5
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=seed)
rfc = RandomForestClassifier()

# Training model.
rfc = rfc.fit(x, y)

# Feature rank.
importance = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
indices = np.argsort(importance)[::-1]
index = 20
indices1 = np.argsort(importance)[::-1][0: index]
print('features rank：%s' % rfc.feature_importances_)
for f in range(100):
    print("%2d) %-*s %f" % (
        f + 1, 30, x.columns[indices[f]], importance[indices[f]]))
raw_data = rfc.predict(xTest).T
pred_data = yTest
ind = np.arange(index)
plt.figure()
plt.title("Feature importances")
plt.bar(range(index), importance[indices1], color="SkyBlue", align="center")
plt.xticks(range(index), x.columns[indices1])
plt.xticks(rotation=90)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()

print('Permutation-based Importance')
perm_importance = permutation_importance(rfc, xTest, yTest)
sorted_idx = perm_importance.importances_mean.argsort()
print(np.std(sorted_idx).T)
