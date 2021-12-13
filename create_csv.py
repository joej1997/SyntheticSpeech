# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 01:42:19 2021

@author: joeja
"""
import numpy as np
import pandas as pd
import librosa
import csv

def getConQ(audio,samplerate):
    C = librosa.cqt(audio, sr=samplerate)
    Real=C.real
    Imaginary=C.imag
    Real=Real.flatten('F')
    Imaginary=Imaginary.flatten("F")
    returnVec=np.concatenate((Real, Imaginary), axis=None)
    return returnVec


def get_mfcc_features(audio_data,sample_rate):
    mfccs = librosa.feature.mfcc(audio_data,n_mfcc=13,sr=sample_rate)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    mfcc_feature_vector = np.concatenate((mfccs,delta_mfccs,delta2_mfccs))
    return mfcc_feature_vector.flatten('F')

def group_delay(sig):
    b = np.fft.fft(sig)
    n_sig = np.multiply(sig, np.arange(len(sig)))
    br = np.fft.fft(n_sig)
    return np.divide(br, b + 0.01).real

def group_delay_coeffs(data):
    N_w = 1024
    L = np.floor(len(data)/N_w).astype(int)

    full_delay = np.zeros((L, N_w))
    for index in range(L):
        window = data[index:(index+N_w)].astype(float)
        full_delay[index][:] = group_delay(window)
    return full_delay.flatten('F')



# import example
data, samplerate = librosa.load('C:/Users/joeja/OneDrive/CPSC554X_project/FakeAvCeleb_dataset/FakeAvCeleb_dataset/drake_av_fake/0.wav',sr=None)
mfcc_vec= get_mfcc_features(data,samplerate)
conQ_vec= getConQ(data,samplerate)
gd_vec = group_delay_coeffs(data)
all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)

#initialize matrix
dat_num=np.zeros((3000,len(all_vec)))
dat_chr = pd.DataFrame(columns=['name', 'dataset', 'interview','isreal',"path","test_train_Part1","test_train_Part2"], index=range(3000))
cur_row=0

import glob
#get drake fake-av celeb data
drake_fake_avc = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/FakeAvCeleb_dataset/FakeAvCeleb_dataset/drake_av_fake/*.wav")
for i in range(len(drake_fake_avc)):
    print(i,"get drake fake-av celeb data")
    data, samplerate = librosa.load(drake_fake_avc[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Drake"
    dat_chr.loc[cur_row,"dataset"]="AVC"
    dat_chr.loc[cur_row,"interview"]="None"
    dat_chr.loc[cur_row,"isreal"]="False"
    dat_chr.loc[cur_row,"path"]= drake_fake_avc[i]
    dat_chr.loc[cur_row,"test_train_Part1"]="None"
    dat_chr.loc[cur_row,"test_train_Part2"]="Test"   
    cur_row+= 1



#get takei fake-av celeb data
takei_fake_avc = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/FakeAvCeleb_dataset/FakeAvCeleb_dataset/takei_av_fake/*.wav")
for i in range(len(takei_fake_avc)):
    print(i,"get takei fake-av celeb data")
    data, samplerate = librosa.load(takei_fake_avc[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Takei"
    dat_chr.loc[cur_row,"dataset"]="AVC"
    dat_chr.loc[cur_row,"interview"]="None"
    dat_chr.loc[cur_row,"isreal"]="False"
    dat_chr.loc[cur_row,"path"]= takei_fake_avc[i]
    dat_chr.loc[cur_row,"test_train_Part1"]= "None"
    dat_chr.loc[cur_row,"test_train_Part2"]= "Test"
    cur_row+= 1



#get drake real data acceptance
drake_real_acceptance = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/drake_acceptance_real/*.wav")
for i in range(len(drake_real_acceptance)):
    print(i,"get drake real data acceptance")
    data, samplerate = librosa.load(drake_real_acceptance[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Drake"
    dat_chr.loc[cur_row,"dataset"]="Real"
    dat_chr.loc[cur_row,"interview"]="Acceptance"
    dat_chr.loc[cur_row,"isreal"]="True"
    dat_chr.loc[cur_row,"path"]= drake_real_acceptance[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Test"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1


#get drake real data grammy
drake_real_grammy = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/drake_grammy_real/*.wav")
for i in range(len(drake_real_grammy)):
    print(i,"get drake real data grammy")
    data, samplerate = librosa.load(drake_real_grammy[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Drake"
    dat_chr.loc[cur_row,"dataset"]="Real"
    dat_chr.loc[cur_row,"interview"]="Grammy"
    dat_chr.loc[cur_row,"isreal"]="True"
    dat_chr.loc[cur_row,"path"]= drake_real_grammy[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Test"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



#get drake real data motivation
drake_real_motivation = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/drake_motivation_real/*.wav")
for i in range(len(drake_real_motivation)):
    print(i, "get drake real data motivation")
    data, samplerate = librosa.load(drake_real_motivation[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Drake"
    dat_chr.loc[cur_row,"dataset"]="Real"
    dat_chr.loc[cur_row,"interview"]="Motivation"
    dat_chr.loc[cur_row,"isreal"]="True"
    dat_chr.loc[cur_row,"path"]= drake_real_motivation[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Test"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get drake tacotron2 data acceptance
drake_fake_acceptance = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/drake_motivation_acceptance_fake/*.wav")
for i in range(len(drake_fake_acceptance)):
    print(i,"get drake tacotron2 data acceptance")
    data, samplerate = librosa.load(drake_fake_acceptance[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Drake"
    dat_chr.loc[cur_row,"dataset"]="Tacotron2"
    dat_chr.loc[cur_row,"interview"]="Acceptance"
    dat_chr.loc[cur_row,"isreal"]="False"
    dat_chr.loc[cur_row,"path"]= drake_fake_acceptance[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get drake tacotron2 data grammy
drake_fake_grammy = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/drake_grammy_fake/*.wav")
for i in range(len(drake_fake_grammy)):
    print(i,"get drake tacotron2 data grammy")
    data, samplerate = librosa.load(drake_fake_grammy[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Drake"
    dat_chr.loc[cur_row,"dataset"]="Tacotron2"
    dat_chr.loc[cur_row,"interview"]="Grammy"
    dat_chr.loc[cur_row,"isreal"]="False"
    dat_chr.loc[cur_row,"path"]= drake_fake_grammy[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get takei real roddenberry
takei_real_roddenberry = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/takei_Roddenberry_real/*.wav")
for i in range(len(takei_real_roddenberry)):
    print(i,"get takei real roddenberry")
    data, samplerate = librosa.load(takei_real_roddenberry[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Takei"
    dat_chr.loc[cur_row,"dataset"]="Real"
    dat_chr.loc[cur_row,"interview"]="Roddenberry"
    dat_chr.loc[cur_row,"isreal"]="True"
    dat_chr.loc[cur_row,"path"]= takei_real_roddenberry[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Test"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get takei real ted
takei_real_ted = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/takei_ted_real/*.wav")
for i in range(len(takei_real_ted)):
    print(i, "get takei real ted")
    data, samplerate = librosa.load(takei_real_ted[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Takei"
    dat_chr.loc[cur_row,"dataset"]="Real"
    dat_chr.loc[cur_row,"interview"]="Ted"
    dat_chr.loc[cur_row,"isreal"]="True"
    dat_chr.loc[cur_row,"path"]= takei_real_ted[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Test"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get takei tacotron2 roddenberry
takei_fake_roddenberry = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/takei_Roddenberry_fake/*.wav")
for i in range(len(takei_fake_roddenberry)):
    print(i, "get takei tacotron2 roddenberry")
    data, samplerate = librosa.load(takei_fake_roddenberry[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Takei"
    dat_chr.loc[cur_row,"dataset"]="Tacotron2"
    dat_chr.loc[cur_row,"interview"]="Roddenberry"
    dat_chr.loc[cur_row,"isreal"]="False"
    dat_chr.loc[cur_row,"path"]= takei_fake_roddenberry[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get takei tacotron2 ted
takei_fake_ted = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/takei_ted_fake/*.wav")
for i in range(len(takei_fake_ted)):
    print(i,"get takei tacotron2 ted")
    data, samplerate = librosa.load(takei_fake_ted[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Takei"
    dat_chr.loc[cur_row,"dataset"]="Tacotron2"
    dat_chr.loc[cur_row,"interview"]="Ted"
    dat_chr.loc[cur_row,"isreal"]="False"
    dat_chr.loc[cur_row,"path"]= takei_fake_ted[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get trump farewell fake
trump_fake_farewell = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/trump_farewell_fake/*.wav")
for i in range(len(trump_fake_farewell)):
    print(i, "get trump farewell fake")
    data, samplerate = librosa.load(trump_fake_farewell[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Trump"
    dat_chr.loc[cur_row,"dataset"]="Tacotron2"
    dat_chr.loc[cur_row,"interview"]="Farewell"
    dat_chr.loc[cur_row,"isreal"]="False"
    dat_chr.loc[cur_row,"path"]= trump_fake_farewell[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get trump inauguration fake
trump_fake_inauguration = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/trump_inauguration_fake/*.wav")
for i in range(len(trump_fake_inauguration)):
    print(i,"get trump inauguration fake")
    data, samplerate = librosa.load(trump_fake_inauguration[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Trump"
    dat_chr.loc[cur_row,"dataset"]="Tacotron2"
    dat_chr.loc[cur_row,"interview"]="Inauguration"
    dat_chr.loc[cur_row,"isreal"]="False"
    dat_chr.loc[cur_row,"path"]= trump_fake_inauguration[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get trump inauguration real
trump_real_inauguration = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/trump_inauguration_real/*.wav")
for i in range(len(trump_real_inauguration)):
    print(i,"get trump inauguration real")
    data, samplerate = librosa.load(trump_real_inauguration[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Trump"
    dat_chr.loc[cur_row,"dataset"]="Real"
    dat_chr.loc[cur_row,"interview"]="Inauguration"
    dat_chr.loc[cur_row,"isreal"]="True"
    dat_chr.loc[cur_row,"path"]= trump_real_inauguration[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Test"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1



# get trump farewell real
trump_real_farewell = glob.glob("C:/Users/joeja/OneDrive/CPSC554X_project/tacotron2_dataset/tacotron2_dataset/trump_farewell_real/*.wav")
for i in range(len(trump_real_farewell)):
    print(i,"get trump farewell real")
    data, samplerate = librosa.load(trump_real_farewell[i],sr=None)
    mfcc_vec= get_mfcc_features(data,samplerate)
    conQ_vec= getConQ(data,samplerate)
    gd_vec = group_delay_coeffs(data)
    all_vec = np.concatenate((mfcc_vec,conQ_vec,gd_vec),axis=None)
    dat_num[cur_row,]=all_vec
    dat_chr.loc[cur_row,'name'] = "Trump"
    dat_chr.loc[cur_row,"dataset"]="Real"
    dat_chr.loc[cur_row,"interview"]="Farewell"
    dat_chr.loc[cur_row,"isreal"]="True"
    dat_chr.loc[cur_row,"path"]= trump_real_farewell[i]
    rand = np.random.uniform(0,1)
    if rand< 0.2:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Test"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Test"
    else:
        dat_chr.loc[cur_row,"test_train_Part1"]= "Train"
        dat_chr.loc[cur_row,"test_train_Part2"]= "Train"
    cur_row+= 1




dat_num=dat_num[0:cur_row,:]
dat_chr = dat_chr.iloc[0:cur_row,:]


with open('dat_num.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write multiple rows
    writer.writerows(dat_num)
    
dat_chr.to_csv("dat_chr.csv")



