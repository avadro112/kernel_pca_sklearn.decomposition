import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read('file.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=0,test_size=0.2)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.decomposition import KernelPCA
kernel = KernelPCA(n_components=2,kernel='rbf')
x_train = kernel.fit_transform(x_train)
x_test = kernel.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
y_pred = classifier.fit(x_train,y_train).predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))