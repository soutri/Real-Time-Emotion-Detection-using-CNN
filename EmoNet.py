
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Input
from keras.models import Model
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,concatenate
from keras import backend as K
from keras.layers import BatchNormalization
from PIL import ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
img_width, img_height = 300,300

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
nos_classes=10
dropout_keep_prob=0.5
# network=Input(shape=[None,img_width,img_height,1])
padding='VALID'
# convolution_initial=Sequential()
# convolution_initial.add(Conv2D(64,(7,7),input_shape=input_shape,padding=padding,strides=2,use_bias=True))
# convolution_initial.add(Activation('relu'))
# convolution_initial.add(MaxPooling2D(pool_size=(3,3)))
# convolution_initial.add(BatchNormalization(axis=-1))
network=Input(shape=input_shape)
# network=Sequential()
conv_1=Conv2D(64,(7,7),padding=padding,strides=2,use_bias=True)(network)
relu_1=Activation('relu')(conv_1)
max_pool_1=MaxPooling2D(pool_size=(3,3),strides=2,padding=padding)(relu_1)
max_pool_1=BatchNormalization()(max_pool_1)
#Feat_Ex_1 Layer
conv_2=Conv2D(96,(1,1),strides=1,padding=padding)(max_pool_1)
conv_2=Activation('relu')(conv_2)
max_pool_2=MaxPooling2D(pool_size=(3,3),strides=1,padding=padding)(max_pool_1)
conv_2_a=Conv2D(208,(3,3),strides=1,padding=padding)(conv_2)
conv_2_a=Activation('relu')(conv_2_a)
conv_2_b=Conv2D(64,(1,1),strides=1,padding=padding)(max_pool_2)
conv_2_b=Activation('relu')(conv_2_b)
feat_ex_1_out=concatenate([conv_2_a,conv_2_b],axis=3)
feat_ex_1_out=feat_ex_1_out
#Feat_Ex_2 Layer
conv_3=Conv2D(96,(1,1),strides=1,padding=padding)(feat_ex_1_out)
conv_3=Activation('relu')(conv_3)
max_pool_3=MaxPooling2D(pool_size=(3,3),strides=1,padding=padding)(feat_ex_1_out)
conv_3_a=Conv2D(208,(3,3),strides=1,padding=padding)(conv_3)
conv_3_a=Activation('relu')(conv_3_a)
conv_3_b=Conv2D(64,(1,1),strides=1,padding=padding)(max_pool_3)
conv_3_b=Activation('relu')(conv_3_b)
feat_ex_2_out=concatenate([conv_3_a,conv_3_b],axis=3)
feat_ex_2_out=MaxPooling2D(pool_size=(3,3),strides=2,padding=padding)(feat_ex_2_out)

#flatten layer
flaten_layer=Flatten()(feat_ex_2_out)
dropout_layer=Dropout(dropout_keep_prob)(flaten_layer)
dense_layer=Dense(nos_classes)(dropout_layer)
dense_layer=Activation('softmax')(dense_layer)
model = Model(inputs=network, outputs=dense_layer)
print(model.summary())
epochs = 10
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.5
# sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
sgd=optimizers.Adagrad(lr=learning_rate,decay=decay_rate)
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
model_config=model.to_json()
with open("model.json",'a') as fileOut:
	fileOut.write(model_config)
#Data ingestion
batch_size = 50
train_data_dir = 'Data/TRAIN'
validation_data_dir = 'Data/validation'
train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
# train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(validation_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode='categorical')
  #  validation_steps=nb_validation_samples // batch_size)
# history=model.fit_generator(train_generator,steps_per_epoch=10,epochs=epochs,validation_data=validation_generator,validation_steps=10)
history=model.fit_generator(train_generator,steps_per_epoch=10,epochs=epochs)
score = model.evaluate_generator(validation_generator,50)
model.save("Classifier_10.h5")


