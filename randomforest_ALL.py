# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 02:27:51 2021

@author: joeja
"""

#import packages
import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as pp
import openpyxl

#import data
df_chr = pd.read_excel("C:/Users/joeja/OneDrive/CPSC554X_project/dat_chr.xlsx")
df_num = pd.read_csv("C:/Users/joeja/OneDrive/CPSC554X_project/dat_num.csv",header=None)
df_num = np.array(df_num)

#Extract Trump data
df_num_trump=df_num #[df_chr["name"]=="Trump",:]
df_chr_trump=df_chr #.loc[df_chr["name"]=="Trump",:]


#split into test (20%) and training data (80%)
where_train = np.where(df_chr_trump["test_train_Part1"]=="Train")[0]
where_test = np.where(df_chr_trump["test_train_Part1"]=="Test")[0]
X_trump_train = df_num_trump[where_train,]
y_trump_train = df_chr_trump.iloc[where_train,4]
X_trump_test = df_num_trump[where_test,]
y_trump_test = df_chr_trump.iloc[where_test,4]

#y_trump_train = np.zeros((len(y_trump_train_chr)))
#for i in range(len(y_trump_train))

#view variance for each variable
sd_vec = np.zeros((X_trump_train.shape[1]))
for i in range(X_trump_train.shape[1]):
    sd_vec[i] = np.std(X_trump_train[:,i])
    X_trump_test[:,i] = X_trump_test[:,i]/sd_vec[i]
    X_trump_train[:,i] = X_trump_train[:,i]/ sd_vec[i]
    

#perform PCA
pca = PCA(n_components= X_trump_train.shape[0])
X_trump_train_pca = pca.fit_transform(X_trump_train)


pp.plot(pca.explained_variance_ratio_) #plot the variance explained and estimate elbow
cutoff=50
pp.plot([cutoff, cutoff],[0,.04]) #plot cutoff

X_trump_test_pca = pca.transform(X_trump_test)

X_trump_train_pca=X_trump_train_pca[:,0:cutoff]
X_trump_test_pca=X_trump_test_pca[:,0:cutoff]


#perform random forests
model = RandomForestClassifier()
model.fit(X_trump_train_pca, y_trump_train)
yhat = model.predict(X_trump_test_pca)


#get test accuracy
sklearn.metrics.confusion_matrix(y_true=y_trump_test, y_pred=yhat)



###############################################################################33
#######################33      Part 2      ######################################
#################################################################################
#Extract Trump data
df_num_trump=df_num #[df_chr["name"]=="Trump",:]
df_chr_trump=df_chr #.loc[df_chr["name"]=="Trump",:]


#split into test (20%) and training data (80%)
where_train = np.where(df_chr_trump["test_train_Part2"]=="Train")[0]
where_test = np.where(df_chr_trump["test_train_Part2"]=="Test")[0]
X_trump_train = df_num_trump[where_train,]
y_trump_train = df_chr_trump.iloc[where_train,4]
X_trump_test = df_num_trump[where_test,]
y_trump_test = df_chr_trump.iloc[where_test,4]

#y_trump_train = np.zeros((len(y_trump_train_chr)))
#for i in range(len(y_trump_train))

#view variance for each variable
sd_vec = np.zeros((X_trump_train.shape[1]))
for i in range(X_trump_train.shape[1]):
    sd_vec[i] = np.std(X_trump_train[:,i])
    X_trump_test[:,i] = X_trump_test[:,i]/sd_vec[i]
    X_trump_train[:,i] = X_trump_train[:,i]/ sd_vec[i]
    

#perform PCA
pca = PCA(n_components= X_trump_train.shape[0])
X_trump_train_pca = pca.fit_transform(X_trump_train)


pp.plot(pca.explained_variance_ratio_) #plot the variance explained and estimate elbow
cutoff=50
pp.plot([cutoff, cutoff],[0,.04]) #plot cutoff

X_trump_test_pca = pca.transform(X_trump_test)

X_trump_train_pca=X_trump_train_pca[:,0:cutoff]
X_trump_test_pca=X_trump_test_pca[:,0:cutoff]


#perform random forests
model = RandomForestClassifier()
model.fit(X_trump_train_pca, y_trump_train)
yhat = model.predict(X_trump_test_pca)


#get test accuracy
sklearn.metrics.confusion_matrix(y_true=y_trump_test, y_pred=yhat)



