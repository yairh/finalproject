import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1,n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

# use Conv2D, MaxPooling2D - twice
# use Conv2D, UpSampling2D - twice
x = Conv2D(32,kernel_size= (28,28), strides=1, padding='same', name='Conv1')(input_img)     # YOUR_CODE )(input_img)
x = MaxPooling2D(pool_size = (2, 2),strides = 2)(x)   # YOUR_CODE)(x)
x = Conv2D(32,kernel_size= (28,28), strides=1, padding='same', name='Conv2')(x)     # YOUR_CODE )(input_img)
x = MaxPooling2D(pool_size = (2, 2))(x)   # YOUR_CODE)(x)
encoded = # YOUR_CODE

# at this point the representation is (7, 7, 32)

x = Conv2D(# YOUR_CODE)(encoded)
x = UpSampling2D(# YOUR_CODE)(x)
x = # YOUR_CODE
x = # YOUR_CODE
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(# YOUR_CODE)
autoencoder.compile(# YOUR_CODE)


autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))


decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


