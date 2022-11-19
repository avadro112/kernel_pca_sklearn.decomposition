import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read('file.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=0,test_size=0.2)
