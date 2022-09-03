import pandas as pd
import shutil
import os

DATA_FOLDER = "data"
labels = pd.read_csv(f"{DATA_FOLDER}/list_attr_celeba.csv")
labels.index = labels["image_id"]


def split_into_folders(feature):
    #copy all images from train folder to train/0 and train/1
    #copy all images from eval folder to eval/0 and eval/1
    #create a folder for each feature
    images = os.listdir("data/train")

    # create directories if they dont exist
    if not os.path.exists(f"data/temp/1"):
        os.makedirs(f"data/temp/1")

    if not os.path.exists(f"data/temp/0"):
        os.makedirs(f"data/temp/0")


    for image in images:
        if labels[feature][image] == 1:
            shutil.copy("data/train/"+image, "data/temp/1/"  + image)
        else:
            shutil.copy("data/train/"+image, "data/temp/0/" + image)

feature = "5_o_Clock_Shadow"
split_into_folders(feature)

# tensorflow dataloader
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255)

train_datagen = train_datagen.flow_from_directory(
        "data/temp",
        target_size=(178, 218),
        batch_size=32,
        class_mode='binary')


import tensorflow as tf
# import tensorflow hub
import tensorflow_hub as hub
#import sequential
from tensorflow.keras import Sequential

model_url = "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1"
module = hub.KerasLayer(model_url, input_shape=(178, 218, 3))
model = Sequential([module])
for layer in model.layers:
    layer.trainable = False
#model.build([None, 178, 218, 3])  # Batch input shape.
fv = model.predict(train_datagen)

#shutil.rmtree("data/temp")