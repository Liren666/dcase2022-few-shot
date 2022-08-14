import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import librosa
from librosa import display as librosadisplay

hdf_path_1 = os.path.join('E:/Pythonproject/dcase-few-shot-bioacoustic/Features/feat_train', 'Mel_train_WMW_P.h5')
#hdf_path_2 = os.path.join('E:/Pythonproject/dcase-few-shot-bioacoustic/Features/feat_train', 'Mel_train_WMW.h5')
hdf_train_1 = h5py.File(hdf_path_1, 'r+')
#hdf_train_2 = h5py.File(hdf_path_2, 'r+')
feat_1 = hdf_train_1['features'][:]
lable_1 = [s.decode() for s in hdf_train_1['labels'][:]]
#lable_2 = [s.decode() for s in hdf_train_2['labels'][:]]
#feat_2 = hdf_train_2['features'][:]

print(len(feat_1))


feat_sample = feat_1[1000]
print("data1",feat_1)
print("label1",lable_1)




D = librosa.amplitude_to_db(np.abs(feat_sample), ref=np.max)
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=22050)

plt.colorbar(img,  format="%+2.f dB")
plt.show()






