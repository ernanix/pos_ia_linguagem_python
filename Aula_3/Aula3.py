# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 09:32:26 2022

@author: ernan
"""

from sklearn import datasets, svm
import matplotlib.pyplot as plt

iris = datasets.load_iris()
digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(digits.data[:-1], digits.target[:-1])
clf.predict(digits.data[-1:]) #[-1:] retorna matriz ao inves de array 


plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
