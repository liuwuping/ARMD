from collections import Counter
from imblearn.over_sampling import ADASYN
import csv
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# Read the data.
raw = []
with open("path") as f:
    reader = csv.reader(f)
    for row in reader:
        raw.append(row)
    raw_data = pd.DataFrame(raw[1:])
raw_features = raw_data.drop([0], axis=1)
features = raw_features.values.astype(np.float64)
classes = raw_data[0].values.astype(np.int64)
print('Original dataset shape %s' % Counter(classes))

# ADASYN algorithm.
ada = ADASYN(random_state=10)
over_sampling_features, over_sampling_classes = ada.fit_resample(features, classes)
print('Resampled dataset shape %s' % Counter(over_sampling_classes))
pd_data5 = pd.DataFrame(over_sampling_features)
pd_data6 = pd.DataFrame(over_sampling_classes)

# balance data.

# features data.
pd_data5.to_csv('E:\path/X.csv')
# classes data.
pd_data6.to_csv('E:\path/y.csv')
