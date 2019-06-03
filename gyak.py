import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
import PIL
from PIL import Image


def create_data(raw):
    data = []
    raw.__delitem__("label")
    for i in range(len(raw.index)):
        snippets = []
        j = 0
        while (j<3584):
            snippet = np.array(raw.values[i,j:j+256])
            j+=256 #bejövő data-tl függ mekkora épp
            snippets.append(snippet)
        data.append(make_ABT_AVGS(snippets))
    return data


def FFT(snippet):
    Fs = 128.0  # sampling rate
    Ts = 2.0/Fs # sampling interval
    t = np.arange(0,1,Ts) # time vector
    y = snippet
    ff = 5  # frequency of the signal
    n = len(y)
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range
    y = y - np.mean(y)
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    return frq,abs(Y)


def make_ABT_AVGS(snippets):
        frame = []
        for i in range(len(snippets)):
            frq, Y = FFT(snippets[i])
            plt.plot(frq, Y)
            plt.show()
            theta, alpha, beta = theta_alpha_beta_averages(frq, Y)
            frame.append([theta, alpha, beta])
        return np.array(frame)

def theta_alpha_beta_averages(f,Y):
    theta_range = (4,8)
    alpha_range = (8,12)
    beta_range = (12,30)
    theta = Y[(f>theta_range[0]) & (f<=theta_range[1])].mean()
    alpha = Y[(f>alpha_range[0]) & (f<=alpha_range[1])].mean()
    beta = Y[(f>beta_range[0]) & (f<=beta_range[1])].mean()
    return theta, alpha, beta



# Make raw_AVG DATA csv-s -> features csv-s files /két részbe kellet memo miatt
# for i in range(6,7):
#     actual_train_raw = pd.read_csv("data_"+(i+1).__str__()+".csv", sep=";")
#     train_labels = actual_train_raw["label"].values
#     print("Beolvasva:",i+1)
#     actual_AVGs = np.array(create_data(actual_train_raw))
#
#     #9et 1re irni ha előröl mennek be adatok
#     print("Features generálva:",i+9)
#     train_all = actual_AVGs.reshape(np.size(actual_AVGs, 0), 42)
#
#     np.savetxt("train_features_"+(i+9).__str__()+".csv", train_all, delimiter=";")
#     np.savetxt("train_labels_"+(i+9).__str__()+".csv", train_labels, delimiter=";")
#     print("Kész:",i+9)
#
# print("KÉSZ KÉSZ KÁSZ KÁSZ KSZ MINDEEEEEEEEEEEEEEEEN!!!!!!!!!!!!!!!!!")
# exit(0)


#Load train csv-s
tf1 = pd.read_csv("train_features_1.csv", sep=";")
tf2 = pd.read_csv("train_features_2.csv", sep=";")
tf3 = pd.read_csv("train_features_3.csv", sep=";")
tf4 = pd.read_csv("train_features_4.csv", sep=";")
tf5 = pd.read_csv("train_features_5.csv", sep=";")
tf6 = pd.read_csv("train_features_6.csv", sep=";")
tf7 = pd.read_csv("train_features_7.csv", sep=";")
tf8 = pd.read_csv("train_features_8.csv", sep=";")
tf9 = pd.read_csv("train_features_9.csv", sep=";")
tf10 = pd.read_csv("train_features_10.csv", sep=";")
tf11 = pd.read_csv("train_features_11.csv", sep=";")
tf12 = pd.read_csv("train_features_12.csv", sep=";")
tf13 = pd.read_csv("train_features_13.csv", sep=";")
tf14 = pd.read_csv("train_features_14.csv", sep=";")
tf15= pd.read_csv("train_features_15.csv", sep=";")
#tf16 = pd.read_csv("train_features_16.csv", sep=";")
#tf17 = pd.read_csv("train_features_17.csv", sep=";")

tl1 = pd.read_csv("train_labels_1.csv", sep=";")
tl2 = pd.read_csv("train_labels_2.csv", sep=";")
tl3 = pd.read_csv("train_labels_3.csv", sep=";")
tl4 = pd.read_csv("train_labels_4.csv", sep=";")
tl5 = pd.read_csv("train_labels_5.csv", sep=";")
tl6 = pd.read_csv("train_labels_6.csv", sep=";")
tl7 = pd.read_csv("train_labels_7.csv", sep=";")
tl8 = pd.read_csv("train_labels_8.csv", sep=";")
tl9 = pd.read_csv("train_labels_9.csv", sep=";")
tl10 = pd.read_csv("train_labels_10.csv", sep=";")
tl11 = pd.read_csv("train_labels_11.csv", sep=";")
tl12 = pd.read_csv("train_labels_12.csv", sep=";")
tl13 = pd.read_csv("train_labels_13.csv", sep=";")
tl14 = pd.read_csv("train_labels_14.csv", sep=";")
tl15 = pd.read_csv("train_labels_15.csv", sep=";")
#tl16 = pd.read_csv("train_labels_16.csv", sep=";")
#tl17 = pd.read_csv("train_labels_17.csv", sep=";")

#Összesítés
data = pd.concat([tf1,tf2,tf3,tf4,tf5,tf6,tf7,tf8,tf9,tf10,tf11,tf12,tf13,tf14,tf15])
labels = pd.concat([tl1,tl2,tl3,tl4,tl5,tl6,tl7,tl8,tl9,tl10,tl11,tl12,tl13,tl14,tl15])

#Data meg label egyesítés
data_and_labels = np.append(labels,data,axis=1)
#Keverés
np.random.shuffle(data_and_labels)

#Szétválasztás
labels = data_and_labels[:,0]
data = data_and_labels[:,1:]

#Normalizásáls
data  = data.values
data = (data - data.min(0)) / data.ptp(0)
labels = labels.reshape([labels.shape[0],1])
split = 5500
#Rename for keras
train_X, test_X = data[:split,:], data[split:,:]
train_Y, test_Y = labels[:split], labels[split:]
#Keras
train_label = keras.utils.to_categorical(train_Y)
test_label = keras.utils.to_categorical(test_Y)
model = keras.Sequential()
model.add(keras.layers.Dense(30, activation=tf.nn.relu,input_shape=(42,)))
model.add(keras.layers.Dense(7, activation=tf.nn.relu))
model.add(keras.layers.Dense(3, activation=tf.nn.softmax))

optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=optim,
              metrics=['accuracy'])
model.summary()

epochs = 5000
batch_size= 64

model.fit(train_X,train_Y,epochs=epochs, batch_size=batch_size)

test_loss, test_acc = model.evaluate(test_X,test_Y)

print('Accuracy:',test_acc, 'Loss:',test_loss)
print("DONE")


#Predict
# print("-----Predict:---- ")
# test0 = pd.read_csv('test0.csv',sep=";")
# test0.drop(['label'], 1)
# classes = model.predict(test0,batch_size=1)
# print(classes)
# if classes.max() == classes[0,0]:
#     print("A szám a 0(nulla)")
# elif classes.max() == classes[0,1]:
#     print("A szám a 1(egy)")
# else:
#     print("A szám a 2(kettő)")
#
# print("-----Predict:----")
# test1 = pd.read_csv('test1.csv', sep=";")
# test1.drop(['label'], 1)
# classes = model.predict(test1, batch_size=32)
# print(classes)
# if classes.max() == classes[0, 0]:
#     print("A szám a 0(nulla)")
# elif classes.max() == classes[0, 1]:
#     print("A szám a 1(egy)")
# else:
#     print("A szám a 2(kettő)")

# print("-----Predict:----")
# test2 = pd.read_csv('test2.csv',sep=";")
# test2.drop(['label'], 1)
# classes = model.predict(test2,batch_size=1)
# print(classes)
# if classes.max() == classes[0,0]:
#     print("A szám a 0(nulla)")
# elif classes.max() == classes[0,1]:
#     print("A szám a 1(egy)")
# else:
#     print("A szám a 2(kettő)")


#
# def gen_images(locs, features, n_gridpoints, normalize=True,
#                augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
#     feat_array_temp = []
#     nElectrodes = locs.shape[0]  # Number of electrodes
#     # Test whether the feature vector length is divisible by number of electrodes
#     assert features.shape[1] % nElectrodes == 0
#     n_colors = features.shape[1] // nElectrodes
#     for c in range(int(n_colors)):
#         feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])
#     nSamples = features.shape[0]
#     # Interpolate the values
#     grid_x, grid_y = np.mgrid[
#                      min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
#                      min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j
#                      ]
#     temp_interp = []
#     for c in range(n_colors):
#         temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
#     # Generate edgeless images
#     if edgeless:
#         min_x, min_y = np.min(locs, axis=0)
#         max_x, max_y = np.max(locs, axis=0)
#         locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
#         for c in range(n_colors):
#             feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
#     # Interpolating
#     for i in range(nSamples):
#         for c in range(n_colors):
#             temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
#                                                method='cubic', fill_value=np.nan)
#         print('Interpolating {0}/{1}\r'.format(i + 1, nSamples), end='\r')
#     # Normalizing
#     for c in range(n_colors):
#         if normalize:
#             temp_interp[c][~np.isnan(temp_interp[c])] = \
#                 scale(temp_interp[c][~np.isnan(temp_interp[c])])
#         temp_interp[c] = np.nan_to_num(temp_interp[c])
#     return np.swapaxes(np.asarray(temp_interp), 0, 1)
#
#
# locs_2d = [(-2.0,4.0),
#            (2.0,4.0),
#            (-1.0,3.0),
#            (1.0,3.0),
#            (-3.0,3.0),
#            (3.0,3.0),
#            (-2.0,2.0),
#            (2.0,2.0),
#            (-2.0,-2.0),
#            (2.0,-2.0),
#            (-4.0,1.0),
#            (4.0,1.0),
#            (-1.0,-3.0),
#            (1.0,-3.0)]
#
# X_0 = make_ABT_AVGS()
# X_1 = X_0.reshape(len(X_0), 14 * 3)
#
# images = gen_images(np.array(locs_2d), X_1, 28, normalize=False)
# images = np.swapaxes(images, 1, 3)
#
# img = Image.fromarray(images[0, :, :, :], 'RGB')
# img.save("brain.png")
#
# print("Done!")
#
# img.show()