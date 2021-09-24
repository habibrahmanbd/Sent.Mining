# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 17:03:30 2017

@author: Habibur Rahman
"""

import os
import numpy as np
#from google.colab import drive

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
import time

h = .02  # step size in the mesh

def make_Corpus(root_dir):
    polarity_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    corpus = []    
  
    for polarity_dir in polarity_dirs:
        reviews = [os.path.join(polarity_dir,f) for f in os.listdir(polarity_dir)]
        for review in reviews:
            doc_string = "";
            with open(review) as rev:
                for line in rev:
                    doc_string = doc_string + line
            if not corpus:
                corpus = [doc_string]
            else:
                corpus.append(doc_string)
    #print "Corpus\n",corpus
    return corpus

  
#drive.mount('/content/drive/')
  
#Create a dictionary of words with its frequency

# root_dir = 'yelp_txt_sentiment'
root_dir = 'txt_sentoken'
corpus = make_Corpus(root_dir)


#Prepare feature vectors per training mail and its labels
labels = np.zeros(2000);
labels[0:1000]=0;
labels[1000:2000]=1; 
      
kf = StratifiedKFold(n_splits=10)



names = ["Logistic Regression", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
         
names = ["Logistic Regression"]
#"Nearest Neighbors", "Linear SVM", "RBF SVM",
#"Nearest Neighbors", "Linear SVM", "RBF SVM",
classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    LogisticRegression(),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

scaler=MinMaxScaler(copy=True, feature_range=(0, 1))

#TSVD   start from here
svd=TruncatedSVD(n_components=1000,n_iter=10,random_state=99)
#svd=NMF(n_components=100)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
#after normalize
vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english', ngram_range=(2,2))
corpus = vectorizer.fit_transform(corpus)

#fit corpus using tsvd
corpus=lsa.fit_transform(corpus)
corpus=scaler.fit_transform(corpus)

for name, model in zip(names, classifiers):
    total = 0
    Y_Total = []
    Y_Pred = []
    totalMat = np.zeros((2,2))
    trainTime = 0.0
    for train_index, test_index in kf.split(corpus,labels):
        y_train, y_test = labels[train_index], labels[test_index]
        X_train = [corpus[i] for i in train_index]
        X_test = [corpus[i] for i in test_index]
        Total_Train_data = len(X_train)
        Total_Test_data = len(X_test)
        st = time.clock()
        model.fit(X_train, y_train)
        en = time.clock()
        trainTime +=(en-st)
        result = model.predict(X_test)
        Y_Total.append(y_test)
        Y_Pred.append(result)
        #totalMat = totalMat + confusion_matrix(y_test, result, labels=[0,1])
        total = total+sum(y_test==result)
        #print classification_report(y_test, result, labels=[0,1])
    '''
    print "\n"+name+":\n";
    print "Confusion matrix:\n",totalMat    
    print "performance:", (total/2000.0)*100.0
    print "Con. Acc: ", ((totalMat[0][0]+totalMat[1][1])/(totalMat[0][0]+totalMat[1][1]+totalMat[0][1]+totalMat[1][0]))
    P = ((totalMat[1][1])/(totalMat[1][1]+totalMat[0][1]))
    R = ((totalMat[1][1])/(totalMat[1][1]+totalMat[1][0]))
    F1= (2.0*((P*R)/(P+R)))
    print "Con. Pre: ", P
    print "Con. Rec: ", R
    print "Con. F1: ", F1
    '''
    #print Y_Total
    #print Y_Pred
    print "\n"+name+":\n";
    print "Confusion matrix:\n",confusion_matrix(np.array(Y_Total).ravel(), np.array(Y_Pred).ravel(), labels=[0,1])
    print "performance:", (total/2000.0)*100.0
    print "Train Time: ", trainTime/10.0
    print classification_report(np.array(Y_Total).ravel(), np.array(Y_Pred).ravel(), labels=[0,1])
    #Y_Total = np.array(Y_Total)
    #Y_Pred = np.array(Y_Pred)
    #print confusion_matrix(Y_Total, Y_Pred)
