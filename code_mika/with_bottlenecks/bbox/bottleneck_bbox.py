import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, DenseNet121



img_width, img_height, channels = 100, 100, 3

train_data_dir = '../imgs/crops/train'
test_data_dir = '../imgs/crops/test'

nb_train_samples = 722
nb_test_samples = 193

batch_size = 16

### Load DenseNet121 model
model = DenseNet121(weights= 'imagenet', include_top=False, input_shape=(img_height, img_width, channels))

### Freeze some layers
for layer in model.layers:
    layer.trainable = False

### Check the trainable status of the individual layers
# for layer in model.layers:
#     print(layer, layer.trainable)

print('**********************TRAIN GENERATOR**********************')
### Train Generator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
                                   )


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size = batch_size,
                                                    class_mode = None,
                                                    shuffle=False)
print('End Train Generator')
# Bottlenecks are the last activation maps before the fully-connected layers in the original model

bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples // batch_size)


print('Ready to save train BF')
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
print('Train BF Saved !')


print('**********************TEST GENERATOR**********************')
### Test Generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size = batch_size,
                                                    class_mode=None,
                                                    shuffle=False)
print('End Test Generator')
bottleneck_features_test = model.predict_generator(test_generator, nb_test_samples // batch_size)
print('Ready to save test BF')
np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)
print('Test BF Saved !')