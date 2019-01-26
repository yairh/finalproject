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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
from glob import glob
import seaborn as sns

print(device_lib.list_local_devices())
print(K.tensorflow_backend._get_available_gpus())
# os.system('sudo chown -R ds:ds /data')
# os.mkdir('output')

# CHOOSE now your model name 
model_name = 'densechest_multilabel'

csvfile = 'data_kaggle/Data_Entry_2017.csv'
df = pd.read_csv(csvfile)

data_dir = '/data/xray_chest_final/'

df = ChestDataset(data_dir,df).reader
df = df[df.exists == True]

df['Finding Labels'] = df['Finding Labels'].replace('No Finding','')
df = df[df['Finding Labels'].isin(list(df['Finding Labels'].value_counts()[:14].index.values))]

labels = list(df['Finding Labels'][~df['Finding Labels'].str.contains('\|')].unique())
labels.remove('')

for label in labels:
    df[label] = df['Finding Labels'].map(lambda x: 1 if label in x else 0)
df['disease_vec'] = df[labels].apply(lambda x: np.array(list(x)),axis=1)
# df.head()

# min_count = np.min(df_uni['Finding Labels'].value_counts())
# df_rd = df_uni.groupby('Finding Labels',group_keys=False).apply(lambda df: df.sample(min_count))

dataset = ChestDataset(data_dir,df)

df['path'] = dataset.image_path

train_list = [el[len(data_dir):] for i,el in enumerate(dataset.image_path) if not i%5 == 0]
test_list = [el[len(data_dir):] for i,el in enumerate(dataset.image_path) if i%5 == 0]

y_train = df['Finding Labels'][df['Image Index'].isin(test_list)]
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

# ADD YOUR MODEL
im_width,im_heigth = 256,256
densenet = DenseNet121(weights='imagenet', include_top=False,input_shape = (im_width,im_heigth,3))

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
# model.add(Dropout(0.248))
model.add(Dense(14, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

train_datagen = ImageDataGenerator(rescale=1./255,
                                   samplewise_center=True, 
                                   samplewise_std_normalization=True,
                                   horizontal_flip = True,
                                   vertical_flip = False, 
                                   height_shift_range= 0.05, 
                                   width_shift_range=0.1, 
                                   rotation_range=5,
                                   shear_range = 0.1,
                                   fill_mode = 'reflect',
                                   zoom_range=0.15)

validation_datagen = ImageDataGenerator(rescale=1./255)

df_train = df[df['Image Index'].isin(train_list)]
df_test = df[df['Image Index'].isin(test_list)]

# Change the batchsize according to your system RAM
train_batchsize = 20
val_batchsize = 20

train_generator = flow_from_dataframe(train_datagen, df_train, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = (im_width,im_heigth),
                             color_mode = 'rgb',
                            batch_size = train_batchsize)

validation_generator = flow_from_dataframe(validation_datagen, df_test, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = (im_width,im_heigth),
                             color_mode = 'rgb',
                            batch_size = val_batchsize)

# Compile the model
optimizer = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='output/logs', histogram_freq=0,
                          write_graph=True, write_images=False)
filepath = "output/checkpoint_{}.hdf5".format(model_name)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    class_weight=class_weights,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=1,
    callbacks=[tensorboard,checkpoint],
    use_multiprocessing=True)

# serialize model to JSON
model_json = model.to_json()
with open("output/{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("output/{}.h5".format(model_name))
print("Saved model to disk")

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
# plt.show()
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
#plt.show()
fig.savefig('output/roc_curve_{}.png'.format(model_name))

print('End Of Training')
