import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from augment import df
from img_convert import convert_image_to_data



X = df.iloc[:, 0:2500]
y = df.iloc[:, -1]

xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=2)

sc = StandardScaler()

xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)



    
clf = SVC(kernel='poly')
clf.fit(xtr, ytr)
joblib.dump(clf, 'svm_model4.joblib')

ypred = clf.predict(xte)

print(confusion_matrix(yte, ypred))
print(accuracy_score(yte, ypred))


# test_image_path = r'C:\python_test\ML_project\img_test\aug_3_download.png'

# image_data = convert_image_to_data(test_image_path)

# if image_data is not None:
#     # The scaler expects a 2D array, so we reshape our 1D array
#     image_data_reshaped = image_data.reshape(1, -1)
    
#     # Scale the new data using the fitted scaler
#     scaled_image_data = sc.transform(image_data_reshaped)
    
#     # Predict using the scaled data
#     test = clf.predict(scaled_image_data)
#     print(test)
# else:
#     print("Failed to read or process the image.")


