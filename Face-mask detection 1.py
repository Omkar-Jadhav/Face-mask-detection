# Import the packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialise the batch size
batch_size=32

#Getting the data
directory=r"dataset"
categories=["with_mask","without_mask"]

#Initializing the list of data and class images
print("loading images...")
data=[]     # Data lists
labels=[]       #Labels list


for category in categories:
    path=os.path.join(directory,category)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path,target_size=(224,224))  #from keras.preprocessing
        image = img_to_array(image)       #Convert image to array
        image = preprocess_input(image)   

        data.append(image)
        labels.append(category)

# Do one hot encoding for the labels. (since initially it were strings)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

#Convert lists into numpy arrays
data=np.array(data,dtype='float32')
labels=np.array(labels)

#Splitting the data 80% train and 20% test
# Stratify keeps the same % of labelled data in both test and train
(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.2,
                                stratify=labels,random_state=42)

#perform data augmentation
aug=ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

#Load the ResNet network, Not including the top layer of the data
baseModel=MobileNetV2(weights="imagenet",include_top=False,
                      input_tensor=Input(shape=(224, 224, 3)))

#Construct the head of the model that will be placed on top of the base model
headmodel = baseModel.output
headmodel = AveragePooling2D(pool_size=(7, 7))(headmodel)
headmodel = Flatten(name="flatten")(headmodel)
headmodel = Dense(128,activation='relu')(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(2,activation='softmax')(headmodel)


#Place the head FC model on top of base model
#Complete model
model = Model(inputs=baseModel.input, outputs=headmodel)


# Freeze the layers in the baseModel
for layer in baseModel.layers:
    layer.trainable=False

#Compiling the model
print(" compiling model...")
optim=Adam(lr=init_LR,decay=init_LR/epochs)
model.compile(loss="binary_crossentropy",optimizer='adam',
              metrics=["accuracy"])

#train the head model
print(" Training head model...")
history=model.fit(aug.flow(trainX,trainY,
             batch_size=batch_size),
             steps_per_epoch=len(trainX)//batch_size,
             validation_data=(testX,testY),
             validation_steps=len(testX)//batch_size,
             epochs=20,verbose=1)

# Make predictions on the testing set
print("Evaluating network..")
predIdxs=model.predict(testX,batch_size=batch_size)

#For each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs=np.argmax(predIdxs,axis=1)

#Shows a nicely forrmatted classification report
print(classification_report(testY.argmax(axis=1),predIdxs,
                            target_names=lb.classes_))
#Serialuze the model to disk
print("Saving the model...")
model.save("Face_mask_detector.model",save_format="h5")

#Plot training loss and accuracy
N=epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arrange(0,N),history.history['loss'],label='train_loss')
plt.plot(np.arrange(0,N),history.history['val_loss'],label='val_loss')
plt.plot(np.arrange(0,N),history.history['accuracy'],label='train_acc')
plt.plot(np.arrange(0,N),history.history['val_accuracy'],label='val_acc')
plt.title("Training loss and accuracy")
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
