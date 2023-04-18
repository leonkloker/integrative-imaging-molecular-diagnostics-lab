#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

tf.keras.backend.set_floatx('float32')


# In[2]:


train_index = ["H-15", "A-15", "B-11", "A-7", "F-9", "G-3", "G-11", "H-7", "I-13"]
path_cores = "TMA_cores_M06_M07_panels/M06/Cores/"
path_mxIF = "Texts_small_coregistered/"


# In[3]:


train_cores = [cv.imread(path_cores + index + ".png") for index in train_index]
train_mxIF = [pd.read_csv(path_mxIF + index + ".csv") for index in train_index]


# In[13]:


BUFFER = 128
BATCH = 32
VAL_SPLIT = 0.04
CELL_SIZE = (32, 32)
MXIF_FEATURES = ["Nucleus PD1 (PPD520) Mean (Normalized Counts, Total Weighting)",
                 "Nucleus PD1 (PPD520) Max (Normalized Counts, Total Weighting)",
                 "Nucleus PD1 (PPD520) Std Dev (Normalized Counts, Total Weighting)",
                 "Nucleus FOXP3 (PPD540) Mean (Normalized Counts, Total Weighting)",
                 "Nucleus FOXP3 (PPD540) Max (Normalized Counts, Total Weighting)",
                 "Nucleus FOXP3 (PPD540) Std Dev (Normalized Counts, Total Weighting)",
                 "Nucleus CD20 (PPD620) Mean (Normalized Counts, Total Weighting)",
                 "Nucleus CD20 (PPD620) Max (Normalized Counts, Total Weighting)",
                 "Nucleus CD20 (PPD620) Std Dev (Normalized Counts, Total Weighting)",
                 "Nucleus CD3 (PPD650) Mean (Normalized Counts, Total Weighting)",
                 "Nucleus CD3 (PPD650) Max (Normalized Counts, Total Weighting)",
                 "Nucleus CD3 (PPD650) Std Dev (Normalized Counts, Total Weighting)",
                 "Nucleus PANCK (PPD690) Mean (Normalized Counts, Total Weighting)",
                 "Nucleus PANCK (PPD690) Max (Normalized Counts, Total Weighting)",
                 "Nucleus PANCK (PPD690) Std Dev (Normalized Counts, Total Weighting)",
                 "Cytoplasm PD1 (PPD520) Mean (Normalized Counts, Total Weighting)",
                 "Cytoplasm PD1 (PPD520) Max (Normalized Counts, Total Weighting)",
                 "Cytoplasm PD1 (PPD520) Std Dev (Normalized Counts, Total Weighting)",
                 "Cytoplasm FOXP3 (PPD540) Mean (Normalized Counts, Total Weighting)",
                 "Cytoplasm FOXP3 (PPD540) Max (Normalized Counts, Total Weighting)",
                 "Cytoplasm FOXP3 (PPD540) Std Dev (Normalized Counts, Total Weighting)",
                 "Cytoplasm CD20 (PPD620) Mean (Normalized Counts, Total Weighting)",
                 "Cytoplasm CD20 (PPD620) Max (Normalized Counts, Total Weighting)",
                 "Cytoplasm CD20 (PPD620) Std Dev (Normalized Counts, Total Weighting)",
                 "Cytoplasm CD3 (PPD650) Mean (Normalized Counts, Total Weighting)",
                 "Cytoplasm CD3 (PPD650) Max (Normalized Counts, Total Weighting)",
                 "Cytoplasm CD3 (PPD650) Std Dev (Normalized Counts, Total Weighting)",
                 "Cytoplasm PANCK (PPD690) Mean (Normalized Counts, Total Weighting)",
                 "Cytoplasm PANCK (PPD690) Max (Normalized Counts, Total Weighting)",
                 "Cytoplasm PANCK (PPD690) Std Dev (Normalized Counts, Total Weighting)"]


# In[14]:


TOTAL_MAX = np.zeros(len(MXIF_FEATURES))
TOTAL_MIN = np.zeros(len(MXIF_FEATURES))

for i, feature in enumerate(MXIF_FEATURES):
    for core in train_mxIF:
        current_max = core.loc[:,feature].max()
        current_min = core.loc[:,feature].min()
        if current_max > TOTAL_MAX[i]:
            TOTAL_MAX[i] = current_max
        if current_min < TOTAL_MIN[i]:
            TOTAL_MIN[i] = current_min


# In[15]:


def get_generator(val=False):
    def data_generator():
        np.random.seed(4)
        for i in range(len(train_index)):
            X = train_mxIF[i].loc[:,'Cell X Position']
            Y = train_mxIF[i].loc[:,'Cell Y Position']

            inx = np.random.uniform(size=X.size) > VAL_SPLIT
            if val:
                inx = np.invert(inx)

            rows = np.arange(X.size)

            for j,x,y in zip(rows[inx],X[inx],Y[inx]):
                x = float(x)
                y = float(y)
                if np.isnan(x) or np.isnan(y):
                    continue
                if round(x - CELL_SIZE[0]) < 0 or round(x + CELL_SIZE[0]) >= train_cores[i].shape[1]:
                    continue
                if round(y - CELL_SIZE[1]) < 0 or round(y + CELL_SIZE[1]) >= train_cores[i].shape[0]:
                    continue

                cell_image = train_cores[i][round(y-CELL_SIZE[1]):round(y+CELL_SIZE[1]),
                                            round(x-CELL_SIZE[0]):round(x+CELL_SIZE[0])] / 255
                
                cell_features = np.array(train_mxIF[i].loc[j, MXIF_FEATURES], dtype=np.float32)
                cell_features = (cell_features - TOTAL_MIN) / TOTAL_MAX
                
                if np.sum(np.isnan(cell_features)) != 0:
                    continue

                yield (cell_image, cell_features), (cell_image, cell_features)
                
    return data_generator


# In[16]:


train_ds = tf.data.Dataset.from_generator(get_generator(),
                    output_signature=((tf.TensorSpec(shape=(2*CELL_SIZE[1],2*CELL_SIZE[0],3), dtype=tf.float32),tf.TensorSpec(shape=(len(MXIF_FEATURES)), dtype=tf.float32)),
                    (tf.TensorSpec(shape=(2*CELL_SIZE[1],2*CELL_SIZE[0],3), dtype=tf.float32),
                    tf.TensorSpec(shape=(len(MXIF_FEATURES)), dtype=tf.float32))))
val_ds = tf.data.Dataset.from_generator(get_generator(val=True),
                    output_signature=((tf.TensorSpec(shape=(2*CELL_SIZE[1],2*CELL_SIZE[0],3), dtype=tf.float32),tf.TensorSpec(shape=(len(MXIF_FEATURES)), dtype=tf.float32)),
                    (tf.TensorSpec(shape=(2*CELL_SIZE[1],2*CELL_SIZE[0],3), dtype=tf.float32),
                    tf.TensorSpec(shape=(len(MXIF_FEATURES)), dtype=tf.float32))))
train_ds = train_ds.shuffle(BUFFER)
train_ds = train_ds.batch(BATCH)
val_ds = val_ds.batch(BATCH)


# In[17]:


class DoubleAutoencoder(tf.keras.models.Model):
    def __init__(self, latent_dim=(64, 16)):
        super(DoubleAutoencoder, self).__init__()
        self.latent_dim_conv = latent_dim[0]
        self.latent_dim_fnn = latent_dim[1]
        
        self.encoder_conv = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2*CELL_SIZE[1], 2*CELL_SIZE[0], 3)),
            tf.keras.layers.Conv2D(filters=self.latent_dim_conv, kernel_size=3, strides=(1, 1), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=self.latent_dim_conv, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=self.latent_dim_conv, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=self.latent_dim_conv, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=self.latent_dim_conv, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=self.latent_dim_conv, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=self.latent_dim_conv, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim_conv)])

        self.decoder_conv = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim_conv)),
            tf.keras.layers.Dense(units=self.latent_dim_conv, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(1, 1, self.latent_dim_conv)),
            tf.keras.layers.Conv2DTranspose(filters=self.latent_dim_conv, kernel_size=3, 
                                            strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=self.latent_dim_conv, kernel_size=3, 
                                            strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=48, kernel_size=3, 
                                            strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, 
                                            strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, 
                                            strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, 
                                            strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=(1, 1), activation='sigmoid', padding='same')])

        self.encoder_fnn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(len(MXIF_FEATURES))),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=self.latent_dim_fnn, activation='relu')
        ])

        self.decoder_fnn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim_fnn)),
            tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer='l2'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer='l2'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=len(MXIF_FEATURES), activation='sigmoid', kernel_regularizer='l2')
        ])

    def call(self, inputs):
        he_latent = self.encoder_conv(inputs[0])
        mxIF_latent = self.encoder_fnn(inputs[1])
        he = self.decoder_conv(he_latent)
        mxIF = self.decoder_fnn(mxIF_latent)
        return he, mxIF


# In[18]:


model = DoubleAutoencoder()
for x, y in val_ds.take(1):
    model(x)
model.summary()


# In[19]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.losses.MeanAbsoluteError()],
)


# In[20]:


log_dir = "logs/autoencoder/baseline_big/"
log_dir_train = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir_train, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss',
                               restore_best_weights=True, patience=20,
                               verbose=0, mode='min')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir_train, monitor='val_loss',
                                verbose = 0, save_best_only=True)


# In[21]:


model.fit(train_ds, validation_data=val_ds, epochs=1000, shuffle=True, verbose=2,
          callbacks=[tensorboard, model_checkpoint])


# In[9]:


loaded_model = tf.saved_model.load(log_dir_train)


# In[12]:


for i, elem in enumerate(val_ds):
    if i == 100:
        predict = loaded_model(elem[0], training=False)
        for j in range(32):
            cv.imwrite("original{}.png".format(j), elem[0][0][j].numpy()*255)
            cv.imwrite("predict{}.png".format(j), predict[0][j].numpy()*255)
        break

