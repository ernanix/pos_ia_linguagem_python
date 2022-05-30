from sklearn import datasets, svm, metrics,naive_bayes


iris = datasets.load_iris()

n_samples = len(iris.data)
data = iris.data

#classifier = svm.SVC(gamma=0.001)
classifier = naive_bayes.GaussianNB()
classifier.fit(data[:n_samples // 2], iris.target[:n_samples // 2])

expected = iris.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
% (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

