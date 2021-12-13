import pandas as pd
import numpy as np
import sys
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as pp

#import data
print("\nImporting data........")
df_chr = pd.read_csv("dat_chr.csv")
df_num = pd.read_csv("dat_num.csv",header=None)
df_num = np.array(df_num)

#Extract Trump data
df_num_trump=df_num[df_chr["name"]=="Trump",:]
df_chr_trump=df_chr.loc[df_chr["name"]=="Trump",:]

#Extract Drake data
df_num_drake = df_num[df_chr["name"]=="Drake",:]
df_chr_drake = df_chr.loc[df_chr["name"]=="Drake",:]

#Extract Takei data
df_num_takei = df_num[df_chr["name"]=="Takei",:]
df_chr_takei = df_chr.loc[df_chr["name"]=="Takei",:]


#split trump into test (20%) and training data (80%)
where_train_trump = np.where(df_chr_trump["test_train_Part2"]=="Train")[0]
where_test_trump = np.where(df_chr_trump["test_train_Part2"]=="Test")[0]
X_trump_train = df_num_trump[where_train_trump,]
y_trump_train = df_chr_trump.iloc[where_train_trump,4]
X_trump_test = df_num_trump[where_test_trump,]
y_trump_test = df_chr_trump.iloc[where_test_trump,4]

#split drake into test (20%) and training data (80%)
where_train_drake = np.where(df_chr_drake["test_train_Part2"]=="Train")[0]
where_test_drake = np.where(df_chr_drake["test_train_Part2"]=="Test")[0]
X_drake_train = df_num_drake[where_train_drake,]
y_drake_train = df_chr_drake.iloc[where_train_drake,4]
X_drake_test = df_num_drake[where_test_drake,]
y_drake_test = df_chr_drake.iloc[where_test_drake,4]

#split takei
where_train_takei = np.where(df_chr_takei["test_train_Part2"]=="Train")[0]
where_test_takei = np.where(df_chr_takei["test_train_Part2"]=="Test")[0]
X_takei_train = df_num_takei[where_train_takei,]
y_takei_train = df_chr_takei.iloc[where_train_takei,4]
X_takei_test = df_num_takei[where_test_takei,]
y_takei_test = df_chr_takei.iloc[where_test_takei,4]


#view variance for each variable
sd_vec_trump = np.zeros((X_trump_train.shape[1]))
for i in range(X_trump_train.shape[1]):
    sd_vec_trump[i] = np.std(X_trump_train[:,i])
    X_trump_test[:,i] = X_trump_test[:,i]/sd_vec_trump[i]
    X_trump_train[:,i] = X_trump_train[:,i]/ sd_vec_trump[i]

sd_vec_drake = np.zeros((X_drake_train.shape[1]))
for i in range(X_drake_train.shape[1]):
    sd_vec_drake[i] = np.std(X_drake_train[:,i])
    X_drake_test[:,i] = X_drake_test[:,i]/sd_vec_drake[i]
    X_drake_train[:,i] = X_drake_train[:,i]/ sd_vec_drake[i]

sd_vec_takei = np.zeros((X_takei_train.shape[1]))
for i in range(X_takei_train.shape[1]):
    sd_vec_takei[i] = np.std(X_takei_train[:,i])
    X_takei_test[:,i] = X_takei_test[:,i]/sd_vec_takei[i]
    X_takei_train[:,i] = X_takei_train[:,i]/ sd_vec_takei[i]

pca = PCA(n_components= X_trump_train.shape[0])
X_trump_train_pca = pca.fit_transform(X_trump_train)


pp.plot(pca.explained_variance_ratio_) #plot the variance explained and estimate elbow
cutoff=25
pp.plot([cutoff, cutoff],[0,.04]) #plot cutoff

X_trump_test_pca = pca.transform(X_trump_test)

X_trump_train_pca=X_trump_train_pca[:,0:cutoff]
X_trump_test_pca=X_trump_test_pca[:,0:cutoff]


#perform random forests
model = RandomForestClassifier(n_estimators=150)
model.fit(X_trump_train_pca, y_trump_train)
yhat = model.predict(X_trump_test_pca)


#get test accuracy
cm_trump = sklearn.metrics.confusion_matrix(y_true=y_trump_test, y_pred=yhat)


print("\nThe confusion matrix for Trump is \n",cm_trump)

pca_drake = PCA(n_components= X_drake_train.shape[0])
X_drake_train_pca = pca_drake.fit_transform(X_drake_train)


pp.plot(pca_drake.explained_variance_ratio_) #plot the variance explained and estimate elbow
cutoff=25
pp.plot([cutoff, cutoff],[0,.04]) #plot cutoff

X_drake_test_pca = pca_drake.transform(X_drake_test)

X_drake_train_pca=X_drake_train_pca[:,0:cutoff]
X_drake_test_pca=X_drake_test_pca[:,0:cutoff]


#perform random forests
model_drake = RandomForestClassifier(n_estimators=150)
model_drake.fit(X_drake_train_pca, y_drake_train)
yhat_drake = model_drake.predict(X_drake_test_pca)


#get test accuracy
cm_drake = sklearn.metrics.confusion_matrix(y_true=y_drake_test, y_pred=yhat_drake)

print("\nThe confusion matrix for drake is \n",cm_drake)

pca_takei = PCA(n_components= X_takei_train.shape[0])
X_takei_train_pca = pca_takei.fit_transform(X_takei_train)


pp.plot(pca_takei.explained_variance_ratio_) #plot the variance explained and estimate elbow
cutoff=25
pp.plot([cutoff, cutoff],[0,.08]) #plot cutoff

X_takei_test_pca = pca_takei.transform(X_takei_test)

X_takei_train_pca=X_takei_train_pca[:,0:cutoff]
X_takei_test_pca=X_takei_test_pca[:,0:cutoff]


#perform random forests
model_takei = RandomForestClassifier()
model_takei.fit(X_takei_train_pca, y_takei_train)
yhat_takei = model_takei.predict(X_takei_test_pca)


#get test accuracy
cm_takei = sklearn.metrics.confusion_matrix(y_true=y_takei_test, y_pred=yhat_takei)
print("\nThe confusion matrix for takei is \n",cm_takei)