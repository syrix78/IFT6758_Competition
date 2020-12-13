#!venv/bin/python
# coding: utf-8

# # Variational Autoencoder (VAE) with Keras

# Modified from code source: https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/#full-vae-code

# ## Model imports

# In[1]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.layers import BatchNormalization, ZeroPadding2D, Cropping2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy, KLD
from tensorflow.keras import backend as K

import numpy as np
#import matplotlib.pyplot as plt

import pandas as pd
import yaml
from sklearn.model_selection import KFold
import PIL.Image
import matplotlib.pyplot as plt

# Following section allow using RTX graphic cards
gpu_devices = tf.config.list_physical_devices('GPU')
for device in gpu_devices: 
    tf.config.experimental.set_memory_growth(device, True)

tf.random.set_seed(6758)

# Load project config
config_file = open("config.yaml", 'r')
config = yaml.load(config_file)

# ## Loading data

# In[2]:


def load_dataset(name, train=True, shuffle=False):
    # Load profiles
    df = pd.read_csv(config["dataset_location"] + "/" + name + ".csv")
    if shuffle:
        df = df.sample(frac=1, random_state=6758).reset_index(drop=True)

    image_path = config["dataset_location"] + "/" + name + "_profile_images/profile_images_"+ name + "/"
    image_list = image_path + df["Profile Image"]
    images = np.array([np.array(PIL.Image.open(image_file)) 
                    for image_file in image_list])

    if train:
        likes = df["Num of Profile Likes"]
    else:
        likes = None

    return images, likes


# ## Data preprocessing

# In[4]:


def prepare(images, likes, *, validation_count=None, validation_split=None):
    # Parse numbers as floats (which presumably speeds up the training process)
    images = images.astype('float32')

    # Normalize data
    images = images / 255

    assert validation_count is not None or validation_split is not None

    if validation_count is None:
        validation_count = np.floor(images.shape[0] * validation_split).astype("int")
    
    if validation_count == 0:
        images_test = None
    else:
        images_test = images[:-validation_count]
        images = images[:validation_count]

    if likes is not None:
        # Apply a log transform on likes
        likes = np.log10(likes + 1)
        
        likes_test = likes[:-validation_count]
        likes = likes[:validation_count]
    else:
        likes_test = None
    
    return images, likes, images_test, likes_test


# ## Creating the predictor

# ### Predictor definition

# In[5]:


# Definition
def create_layers(input_shape):
    # Define padding required to get a multiple of 8 as dimensions 
    # (for performances on tensor cores)
    input_padding = 0
    v_pad = input_shape[1] % 8
    h_pad = input_shape[2] % 8

    if v_pad != 0 or h_pad != 0:
        top_pad = v_pad // 2
        left_pad = h_pad // 2
        input_padding = ((top_pad, v_pad - top_pad), (left_pad, h_pad - left_pad))
    
    i = Input(shape=input_shape[1:], name='predictor_input')
    cx = ZeroPadding2D(padding=input_padding)(i)
    cx = Conv2D(filters=8, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer = 'he_normal')(cx)
    cx = BatchNormalization()(cx)
    cx = Conv2D(filters=16, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer = 'he_normal')(cx)
    cx = BatchNormalization()(cx)
    #cx = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer = 'he_normal')(cx)
    #cx = BatchNormalization()(cx)
    x = Flatten()(cx)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu', kernel_initializer = 'he_normal')(x)
    x = Dense(1)(x)
    return i, cx, x


# ### Predictor instantiation

# In[7]:

@tf.function
def RMSLE(true, pred):
    return tf.math.sqrt(keras.losses.MSLE(true, pred))

def create_model(input_layer, output_layer):
    # Instantiate predictor
    predictor = Model(input_layer, output_layer, name='predictor')
    #predictor.summary()
    #keras.utils.plot_model(predictor, 'broken_deep_model.png', show_shapes=True)

    # Compile with tf optimiser to use tensor cores
    opt = tf.keras.optimizers.Adam()
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    predictor.compile(optimizer=opt, loss=RMSLE, experimental_run_tf_function=False)

    return predictor


#%%


images, likes = load_dataset("train", train=True, shuffle=True)
images, likes, images_test, likes_test = prepare(images, likes, validation_count=1000)

# Create layers
i, _, x = create_layers(images.shape)

# Create model
predictor = create_model(i, x)


# ## Training

# In[13]:


# Train predictor
if config["CV"]:
    validation_losses = []
    for train_index, test_index in KFold(config["n_split"]).split(likes):
        x_train, x_test = images[train_index], images[test_index]
        y_train, y_test = likes[train_index], likes[test_index]
        
        predictor = create_model(i, x)
        predictor.fit(images, likes, epochs=20, validation_split=0.2) 
        validation_losses.append(predictor.evaluate(x_test, y_test))

    print("Validation loss:" + str(np.array(validation_losses).mean()))
else:
    history = predictor.fit(images, likes, epochs=20, validation_split=0.2)  


# ## Visualizing predictor results

# In[15]:


import seaborn as sns; sns.set()

def plotLoss(history):
    loss = history.history["loss"][:1000]
    val_loss = history.history["val_loss"][:1000]
    
    n = 10
    epochs = range(len(loss))[:-n]

    loss_temp = np.zeros((n, len(epochs)))
    val_loss_temp = np.zeros((n, len(epochs)))

    for i in range(n):
        loss_temp[i] = loss[i:i - n]
        val_loss_temp[i] = val_loss[i:i - n]

    loss_mean = loss_temp.mean(axis=0)
    val_loss_mean = val_loss_temp.mean(axis=0)

    loss_std = loss_temp.std(axis=0) * 2
    val_loss_std = val_loss_temp.std(axis=0) * 2

    upper_loss = loss_mean + 2 * loss_std
    lower_loss = loss_mean - 2 * loss_std

    upper_val_loss = val_loss_mean + 2 * val_loss_std
    lower_val_loss = val_loss_mean - 2 * val_loss_std

    fig, ax = plt.subplots(figsize=(12,8))
    clrs = sns.color_palette("deep", 2)
    #with sns.axes_style("darkgrid"):
    ax.plot(range(len(loss)), loss, label="Training loss", c=clrs[0])
    #ax.fill_between(epochs, lower_loss, upper_loss, alpha=0.3 ,facecolor=clrs[0])
    ax.plot(range(len(val_loss)), val_loss, label="Validation loss", c=clrs[1])
    #ax.fill_between(epochs, lower_val_loss, upper_val_loss, alpha=0.3 ,facecolor=clrs[1])
    ax.set_xlabel("Epoch")
    ax.legend()

    plt.show()

# In[ ]


plotLoss(history)
# %%


predictor.summary()


#%%

predictions_log = predictor.predict(images_test)
predictions = np.power(10, predictions_log)

# %%

log_y = np.log10(likes_test + 1)
log_y_hat = np.log10(predictions + 1).reshape(-1,)

error = np.sqrt((np.square(log_y_hat - log_y)).sum() / len(log_y))
print(error)
# %%


# Predict competition
images_comp, _ = load_dataset("test", train=False, shuffle=False)
images_comp, _, _, _ = prepare(images_comp, None, validation_split=0)

predictions_log_comp = predictor.predict(images_comp)
predictions_comp = np.power(10, predictions_log_comp).reshape((-1,))
predictions_comp = np.round(predictions_comp).astype("int")
# %%

df = pd.read_csv(config["dataset_location"] + "/" + "test" + ".csv")
df_submitted = pd.DataFrame({"Id" : df["Id"], "Predicted" : predictions_comp})
# %%
df_submitted.to_csv("submission.csv", index=False)
# %%
