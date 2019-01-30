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
from sklearn.utils import class_weight
import matplotlib as plt
matplotlib.use('Agg')
from keras.optimizers import Adam
import os
from itertools import chain
from collections import Counter
from glob import glob
import tensorflow as tf
import keras
print(device_lib.list_local_devices())
print(K.tensorflow_backend._get_available_gpus())
os.system('sudo chown -R ds:ds /data')
os.mkdir('output')

# CHOOSE now your model name 
model_name = 'densechest_multiclass'

csvfile = 'data_kaggle/Data_Entry_2017.csv'
df = pd.read_csv(csvfile)

data_dir = '/data/xray_chest_final/'

ChestDataset(data_dir,df).reset_folder()

df_uni = ChestDataset(data_dir,df[~df['Finding Labels'].str.contains('\|')]).reader
df_uni = df_uni[df_uni.exists == True]

dataset = ChestDataset(data_dir,df_uni)

train_list = [el[len(data_dir):] for i,el in enumerate(dataset.image_path) if not i%5 == 0]
test_list = [el[len(data_dir):] for i,el in enumerate(dataset.image_path) if i%5 == 0]
class_weights = dict(df_uni.groupby('Finding Labels').count().exists/(len(df_uni)))



with open('output/{}_train_list.txt'.format(model_name), 'w') as f:
    for item in train_list:
        f.write("%s\n" % item)

with open('output/{}_test_list.txt'.format(model_name), 'w') as f:
    for item in test_list:
        f.write("%s\n" % item)

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
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))

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
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    test_folder,
    target_size=(img_height, img_width),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

class_weights = {train_generator.class_indices[k]:v for k,v in class_weights.items()}



# Compile the model
optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='output/logs', histogram_freq=0,
                          write_graph=True, write_images=False)
filepath = "output/checkpoint_{}.hdf5".format(model_name)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=5,
    class_weight=class_weights,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=1,
    callbacks=[tensorboard,checkpoint])

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


preds = np.argmax(prediction,axis=1)

y_true = np.zeros((preds.shape[0],validation_generator.num_classes))
y_true[np.arange(preds.shape[0]), validation_generator.classes] = 1
inv_map = {v:k for k,v in validation_generator.class_indices.items()}
pred_cat = [inv_map[i] for i in preds]

report = classification_report(validation_generator.classes,preds)
np.save('output/report_{}.npy'.format(model_name),report)
print(report)
print('Accuracy score: ',accuracy_score(validation_generator.classes,preds))


score = model.evaluate_generator(validation_generator,
                                 steps=len(validation_generator),
                                 pickle_safe=True)
print('Accuracy Keras: ', score[1])



# Auc scores
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(validation_generator.num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], prediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig = plt.figure(figsize=(15,10))
for i in range(validation_generator.num_classes):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(inv_map[i], roc_auc[i]))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
fig.savefig('output/roc_curve_{}.png'.format(model_name))

precision = dict()
recall = dict()
for i in range(len(labels)):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], prediction[:, i])

fig = plt.figure(figsize=(15,10))
for i in range(len(labels)):
    plt.plot(recall[i], precision[i],
             label='PR curve for {0}'
             ''.format(labels[i]))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve')
plt.legend(loc="lower right")
fig.savefig('output/pr_curve_{}.png'.format(model_name))

print('End Of Training')

