import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
clf = svm.SVC(gamma=.001 , C=100)
x,y = digits.data[:-1], digits.target[:-1]
clf.fit(x,y)

a = 100  #element want to check
print("Prediction: ", clf.predict(digits.data[a])) #for prediction
#true value 
plt.imshow(digits.images[a],cmap=plt.cm.gray_r,interpolation = "nearest")
plt.show()
