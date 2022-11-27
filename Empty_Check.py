import os
import numpy as np
import pickle

from skimage.io import imread
from skimage.transfer import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Prepare data

# location of the data file
input_dir = "C:/CONY/Myclass/OpenCV_python/ParkingSpot/Picture"
categories = ['empty','not_empty']
data = []
labels = []
for category_index, category in enumerate(categories):
	for file in os.listdir(os.path.join(input_dir, category)):
		img_path = os.path.join(input_dir, category, file)
		img = imread(img_path)
		img = resize(img, (15,15))
		data.append(img.flatten())
		labels.append(category_index)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle = True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01,0.001,0.0001], 'C': [1,10,100,1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)


# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score*100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))