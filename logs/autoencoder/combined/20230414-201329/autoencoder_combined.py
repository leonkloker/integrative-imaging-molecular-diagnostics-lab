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
print("GPUs: {}".format(len(tf.config.list_physical_devices("GPU"))))


# In[2]:


train_index = ["H-15", "A-15", "B-11", "A-7", "F-9", "G-3", "G-11", "H-7", "I-13"]
path_cores = "TMA_cores_M06_M07_panels/M06/Cores/"
path_mxIF = "Texts_small_coregistered/"


# In[3]:


train_cores = [cv.imread(path_cores + index + ".png") for index in train_index]
train_mxIF = [pd.read_csv(path_mxIF + index + ".csv") for index in train_index]


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[9]:


autoencoder_separate = tf.saved_model.load("logs/autoencoder/baseline_64dim/ffn_dropout/20230314-220323/")


# In[12]:


class CombinedAutoencoder(tf.keras.models.Model):
    def __init__(self, latent_dim=(80,20), dropout=(0,0)):
        super(CombinedAutoencoder, self).__init__()
        self.autoencoder_separate = autoencoder_separate
        self.latent_dim_pre = latent_dim[0]
        self.latent_dim = latent_dim[1]
        self.dropout_encoder = dropout[0]
        self.dropout_decoder_ffn = dropout[1]

        self.concat = tf.keras.layers.Concatenate(axis=-1)
        
        self.combined_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim_pre)),
            tf.keras.layers.Dense(60),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.dropout_encoder),
            tf.keras.layers.Dense(40),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.dropout_encoder),
            tf.keras.layers.Dense(self.latent_dim, activation='tanh')])

        self.decoder_conv = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim)),
            tf.keras.layers.Dense(units=self.latent_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=48, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=64, activation='relu'),

            tf.keras.layers.Reshape(target_shape=(1, 1, 64)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, 
                                            strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, 
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

        self.decoder_ffn = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim)),
            tf.keras.layers.Dense(units=self.latent_dim, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.dropout_decoder_ffn),
            tf.keras.layers.Dense(units=self.latent_dim),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.dropout_decoder_ffn),
            tf.keras.layers.Dense(units=self.latent_dim),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.dropout_decoder_ffn),
            tf.keras.layers.Dense(units=len(MXIF_FEATURES)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Dropout(self.dropout_decoder_ffn),
            tf.keras.layers.Dense(units=len(MXIF_FEATURES), activation='sigmoid')
        ])

    def call(self, inputs):
        he_latent = self.autoencoder_separate.encoder_conv(inputs[0], training=False)
        mxIF_latent = self.autoencoder_separate.encoder_fnn(inputs[1], training=False)
        h = self.concat([he_latent, mxIF_latent])
        z = self.combined_encoder(h)
        he = self.decoder_conv(z)
        mxIF = self.decoder_ffn(z)
        return he, mxIF
        


# In[13]:


model = CombinedAutoencoder()
for x, y in val_ds.take(1):
    model(x)
model.summary()


# In[14]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.losses.MeanAbsoluteError()],
)


# In[9]:


log_dir = "logs/autoencoder/combined/"
log_dir_train = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir_train, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss',
                               restore_best_weights=True, patience=20,
                               verbose=0, mode='min')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir_train, monitor='val_loss',
                                verbose = 0, save_best_only=True)


# In[21]:


model.fit(train_ds, validation_data=val_ds, epochs=200, shuffle=True, verbose=2,
          callbacks=[tensorboard, model_checkpoint])


# In[13]:


#loaded_model = tf.saved_model.load("logs/autoencoder/baseline_big/20230314-085415/")


# In[1]:


"""for i, elem in enumerate(val_ds):
    if i == 40:
        predict = loaded_model(elem[0], training=False)
        for j in range(32):
            cv.imwrite("original{}.png".format(j), elem[0][0][j].numpy()*255)
            cv.imwrite("predict{}.png".format(j), predict[0][j].numpy()*255)
        break"""

