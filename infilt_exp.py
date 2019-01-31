from class_dataset import ChestDataset
import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.applications import DenseNet121
from keras import models
from keras import backend as K
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
from glob import glob
from sklearn.metrics import roc_curve,auc, classification_report, accuracy_score, precision_recall_curve

print(device_lib.list_local_devices())
print(K.tensorflow_backend._get_available_gpus())
os.system('sudo chown -R ds:ds /data')
os.mkdir('output')

# CHOOSE now your model name 
model_name = 'densechest_infilt'

csvfile = 'data_kaggle/Data_Entry_2017.csv'
df = pd.read_csv(csvfile)

data_dir = '/data/xray_chest_final/'

ChestDataset(data_dir,df).reset_folder()

df = ChestDataset(data_dir,df).reader
df = df[df.exists == True]

train_path = './output/densechest_infilt_train_list.txt'
test_path = './output/densechest_infilt_test_list.txt'
with open(train_path,'r') as f:
    train_list = f.read().split('\n')

with open(test_path,'r') as f:
    test_list = f.read().split('\n')
    
dataset = ChestDataset(data_dir,df)

train_dt,test_dt = dataset.train_test(train_list,test_list)
train_dt.create_tree()
test_dt.create_tree()

train_files = train_dt.image_path
test_files = test_dt.image_path
train_folder = train_dt.dir
test_folder = test_dt.dir


# ADD YOUR MODEL
img_width,img_height = 256,256
densenet = DenseNet121(weights='imagenet', include_top=False,input_shape = (img_width, img_height, 3))

# # Freeze some layers
# for layer in densenet.layers[:100]:
#     layer.trainable = False
    
# Create the model
model = models.Sequential()

model.add(densenet)

# Add new layers
model.add(Flatten())
# model.add(Dense(72))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.248))
model.add(Dense(1, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                  zoom_range=0.2,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_batchsize = 10
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(img_height, img_width),
    batch_size=train_batchsize,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    test_folder,
    target_size=(img_height, img_width),
    batch_size=val_batchsize,
    class_mode='binary',
    shuffle=False)


# Compile the model

optimizer = Adam()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
weight_path = './output/checkpoint_densechest_infilt.hdf5'
try: 
    model.load_weights(weight_path)
except:
    pass

tensorboard = TensorBoard(log_dir='output/logs', histogram_freq=0,
                          write_graph=True, write_images=False)
filepath = "output/checkpoint_{}.hdf5".format(model_name)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=1,
    callbacks=[tensorboard,checkpoint])

# serialize model to JSON
model_json = model.to_json()
with open("output/{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("output/{}.h5".format(model_name))
print("Saved model to disk")


#metrics
fig = plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig('output/history_{}.png'.format(model_name))

prediction = model.predict_generator(validation_generator,
                                     steps=len(validation_generator),
                                     pickle_safe=True,
                                     verbose=1)


preds = np.round(prediction)
report = classification_report(validation_generator.classes,preds)
np.save('output/report_{}.npy'.format(model_name),report)
print(report)
print('Accuracy score: ',accuracy_score(validation_generator.classes,preds))

score = model.evaluate_generator(validation_generator,
                                 steps=len(validation_generator),
                                 pickle_safe=True)
print('Accuracy Keras: ', score[1])

print('End of Training')