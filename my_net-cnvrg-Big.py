
# coding: utf-8

# In[ ]:


from class_dataset import ChestDataset
import pandas as pd
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.applications import DenseNet121
from keras import models
from keras import backend as K
from tensorflow.python.client import device_lib
from glob import glob


# In[ ]:


print(device_lib.list_local_devices())
print(K.tensorflow_backend._get_available_gpus())


# In[ ]:


get_ipython().system('sudo chown -R ds:ds /data')


# In[2]:


csvfile = 'data_kaggle/Data_Entry_2017.csv'
df = pd.read_csv(csvfile)

data_dir = '/data/xray_chest_final/'

ChestDataset(data_dir,df).reset_folder()

df_uni = ChestDataset(data_dir,df[~df['Finding Labels'].str.contains('\|')]).reader
df_uni = df_uni[df_uni.exists == True]

min_count = min(df_uni['Finding Labels'].value_counts())

df_rd = df_uni.groupby('Finding Labels',group_keys=False).apply(lambda df: df.sample(min_count))

dataset = ChestDataset(data_dir,df_rd)

train_list = [el[23:] for i,el in enumerate(dataset.image_path) if not i%5 == 0]
test_list = [el[23:] for i,el in enumerate(dataset.image_path) if i%5 == 0]

with open('dense_train_list.txt', 'w') as f:
    for item in train_list:
        f.write("%s\n" % item)

with open('dense_test_list.txt', 'w') as f:
    for item in test_list:
        f.write("%s\n" % item)

train_dt,test_dt = dataset.train_test(train_list,test_list)
train_dt.create_tree()
test_dt.create_tree()


# In[7]:


train_files = train_dt.image_path
test_files = test_dt.image_path
train_folder = train_dt.dir
test_folder = test_dt.dir


# In[8]:


label_train = [train_dt.labels[i] for i, el in enumerate(train_dt.exists) if el == True]
label_test = [test_dt.labels[i] for i, el in enumerate(test_dt.exists) if el == True]
print('Train # No Finding:',label_train.count('No Finding')/len(label_train))
print('Test # No Finding:',label_test.count('No Finding')/len(label_test))


# In[9]:


labels = set(dataset.labels)
print('Statistics about the Dataset:\n')
print('There are %d total chest deseases.' % len(set(dataset.labels)))
print('There are %s total chest images.\n' % np.sum(dataset.exists))
print('There are %d training chest images.' % np.sum(train_dt.exists))
# print('There are %d validation dog images.' % len(valid_files))
print('There are %d test chest images.'% np.sum(test_dt.exists))
for lab in labels:
    print('# of %s: %.3f%%'%(lab,100*dataset.labels.count(lab)/len(dataset.labels)))


# In[10]:


img_width,img_height = 365,365
densenet = DenseNet121(weights='imagenet', include_top=False,input_shape = (img_width, img_height, 3))

# # Freeze some layers
# for layer in densenet.layers[:]:
#     layer.trainable = False
    
# Create the model
model = models.Sequential()

# Add the vgg convolutional base model

model.add(densenet)

# Add new layers
model.add(Flatten())
# model.add(Dense(72))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.248))
model.add(Dense(15, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()


# In[ ]:


train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

# Change the batchsize according to your system RAM
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

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=1,
    use_multiprocessing=True)


# In[ ]:


prediction = model.predict_generator(validation_generator,
                                     steps=len(validation_generator),
                                     pickle_safe=True,
                                     verbose=1)


# In[ ]:


preds = np.argmax(prediction,axis=1)
print(preds.shape)

y_true = np.zeros((preds.shape[0],validation_generator.num_classes))
y_true[np.arange(preds.shape[0]), validation_generator.classes] = 1
inv_map = {v:k for k,v in validation_generator.class_indices.items()}
pred_cat = [inv_map[i] for i in preds]

print(classification_report(validation_generator.classes,preds))
print('Accuracy score: ',accuracy_score(validation_generator.classes,preds))


# In[ ]:


score = model.evaluate_generator(validation_generator,
                                 steps=len(validation_generator),
                                 pickle_safe=True,
                                 verbose=1)
print('Accuracy Keras: ', score[1])


# In[ ]:


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
fig.savefig('roc_curve.png')

print('end')

