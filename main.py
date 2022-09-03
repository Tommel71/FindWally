import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from torch.functional import F
from sklearn.model_selection import train_test_split
from PIL import Image
import ssl
import os
import pandas as pd
from fastai.vision.all import *
ssl._create_default_https_context = ssl._create_unverified_context



NUM_OUTPUTS = 16
ORIGINAL_IMG_SIZE = 64
EFFICIENT_NET_INPUT_IMG_SIZE = 224
IMG_CHANNELLS = 3



DATA_FOLDER = "data"
EVAL_FILE = f"{DATA_FOLDER}/eval_data_public.csv"
TRAIN_FILE = f"{DATA_FOLDER}/train_data.csv"
IMG_FOLDER = f"{DATA_FOLDER}/img_align_celeba/"

eval_df = pd.read_csv(EVAL_FILE)
train_df = pd.read_csv(TRAIN_FILE)

def show_list_of_images(img_ids_list):
    images = [Image.open(f'{IMG_FOLDER}{img_id}') for img_id in img_ids_list]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    display(new_im)


# Select hardware: be sure to run the model on a GPU!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RUNNING ON:", device)

# Image transformation
# when having a PIL image "im" loaded, you can transform it using the command "transform(im)"
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 128], output_size=NUM_OUTPUTS, dropout=0.2):
        super(Feedforward, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_sizes[1], output_size),
        )

    def forward(self, x):
        return self.layers(x)


from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)



labels = pd.read_csv(f"{DATA_FOLDER}/list_attr_celeba.csv")
feature_name = "5_o_Clock_Shadow"




labels.index = labels["image_id"]
train_names = os.listdir(f'{DATA_FOLDER}/train')
train_labels = labels["5_o_Clock_Shadow"][train_names]
train_labels = list(train_labels)
"""
train_generator = ImageDataLoaders.from_lists(f"data/train/", train_names, train_labels)

labels.index = labels["image_id"]
validation_names = os.listdir(f'{DATA_FOLDER}/validation')
validation_labels = labels["5_o_Clock_Shadow"][validation_names]
validation_labels = list(validation_labels)

validation_generator = ImageDataLoaders.from_lists (f"{DATA_FOLDER}/validation", [f"{DATA_FOLDER}/validation/{t}" for t in validation_names], validation_labels)
"""


#path = untar_data(URLs.PETS)
#fnames = get_image_files(path/"images")
#labels = ['_'.join(x.name.split('_')[:-1]) for x in fnames]
#dls = ImageDataLoaders.from_lists(path, fnames, labels)

path = "data"
train_labels = np.asarray(train_labels)
train_labels[train_labels == -1] = 0
df = pd.DataFrame.from_dict({"name": ["train/" + t for t in train_names], "label":list(train_labels)})
train_loader = ImageDataLoaders.from_df(df, path)


learn = vision_learner(train_loader, resnet34, metrics=error_rate)
learn = vision_learner(train_loader, models.resnet18, loss_func=CrossEntropyLossFlat(), ps=0.25)

learn.fine_tune(1)

