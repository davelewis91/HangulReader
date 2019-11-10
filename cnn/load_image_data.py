import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn.model_selection as sk

import pandas as pd

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, '../images/image-label-map.csv')

def load_labels(label_file):
    file_labels = pd.read_csv(label_file, encoding='utf-16')
    print(file_labels.columns)
    file_labels['labels'] = file_labels['labels'].apply(lambda x: x.split('|'))
    unique_labels = file_labels['labels'].apply(lambda x: set(x))
    unique_labels = {i for v in unique_labels for i in v}
    return file_labels, unique_labels

def make_keras_dataset(sample, generator, unique_labels):
    dataset = generator.flow_from_dataframe(
        sample,
        x_col='file_name',
        y_col='labels',
        batch_size=100,
        seed=1234,
        shuffle=True,
        class_mode='categorical',
        classes=unique_labels
    )

def train_test_split(labels, classes, stratify=True):
    """
    Returns 70-20-10 train/test/validate split
    :param labels: pd dataframe, image paths + labels
    :param stratify: Boolean, default=True, use label_index-based sampling
    :return: keras ImageDataGenerator objects for train/test/validate
    """
    datagen = ImageDataGenerator(rescale=1/255)

    train, test = sk.train_test_split(
        labels,
        train_size=0.7,
        random_state=1234,
        stratify=labels['label_index'] if stratify else None
    )
    test, validate = sk.train_test_split(
        test,
        train_size=0.66,
        random_state=1234,
        stratify=test['label_index'] if stratify else None
    )

    train = make_keras_dataset(train, datagen, classes)
    test = make_keras_dataset(test, datagen, classes)
    validate = make_keras_dataset(validate, datagen, classes)
    return train, test, validate

def load_images(label_file):
    labels, classes = load_labels(label_file)
    return train_test_split(labels, classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing image filepaths and their labels')
    args = parser.parse_args()
    train, test, validate = load_images(args.label_file)

