#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing


def main(fnametr, fnamevl):

        # loads data
        print ("Loading training...")
        data = np.loadtxt(fnametr)
        X_train = data[:, 1:]
        y_train = data[:,0]


        print ("Loading validation...")
        data = np.loadtxt(fnamevl)
        X_test = data[:, 1:]
        y_test = data[:,0]

        # cria um kNN
        neigh = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

        print ('Fitting knn')
        neigh.fit(X_train, y_train)

        # predicao do classificador
        print ('Predicting...')
        y_pred = neigh.predict(X_test)

        # mostra o resultado do classificador na base de teste
        print ('Accuracy: ',  neigh.score(X_test, y_test))

        # cria a matriz de confusao
        cm = confusion_matrix(y_test, y_pred)
        print (cm)
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
        if len(sys.argv) != 3:
                sys.exit("Use: knn.py <datatr> <datats>")

        main(sys.argv[1], sys.argv[2])