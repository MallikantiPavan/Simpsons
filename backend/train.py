import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import caer
import gc
import canaro
import joblib
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical

path = r'C:\Users\pavan\Downloads\simpsons_dataset\simpsons_dataset'
channels = 1 
img_size = (80, 80)
BATCH_SIZE = 32
EPOCHS = 10

char_dic = {}
for char in os.listdir(path):
    char_dic[char] = len(os.listdir(os.path.join(path, char)))

char_dic = caer.sort_dict(char_dic, descending=True)

character = []
count = 0
for char_name, _ in char_dic:
    character.append(char_name)
    count += 1
    if count >= 100:
        break

print("Top characters:", character)

train = caer.preprocess_from_dir(path, character, channels=channels, IMG_SIZE=img_size, isShuffle=True)

plt.figure(figsize=(5,5))
plt.imshow(train[0][0], cmap='gray')

featureset, labels = caer.sep_train(train, IMG_SIZE=img_size)
featureset = caer.normalize(featureset)
labels = to_categorical(labels, len(character))

xtrain, xtest, ytrain, ytest = caer.train_val_split(featureset, labels, val_ratio=0.2)


del train, featureset, labels
gc.collect()

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(xtrain, ytrain, batch_size=BATCH_SIZE)

model = canaro.models.createSimpsonsModel(
    IMG_SIZE=img_size,
    channels=channels,
    output_dim=len(character),
    loss='categorical_crossentropy', 
    decay=1e-6,
    learning_rate=0.001,
    momentum=0.9,
    nesterov=True
)
model.summary()

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(
    train_gen,
    steps_per_epoch=len(xtrain)//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(xtest, ytest),
    callbacks=callbacks_list
)
predictions = model.predict(xtest)
print("Predicted character:", character[np.argmax(predictions[0])])

model.save("simpsons_model.h5")
joblib.dump(character, "labels.pkl")
print(" Model and labels saved")
