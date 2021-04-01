from tqdm import tqdm
import time
import os
import numpy as np
import cv2
from random import shuffle
import pickle


class DataPreProcessor:
    DATA_DIR = None
    IMG_SIZE = None
    DATA_SAVE_PATH = None
    processed_data = []
    categories = []
    description = None

    def __init__(self, options):
        self.DATA_DIR = options['DATA_DIR']
        self.IMG_SIZE = options['IMG_SIZE']
        self.DATA_SAVE_PATH = options['DATA_SAVE_PATH']
        self.description = options['DESCRIPTION']
        self.categories = options['CATEGORIES']

    def start(self):
        # load training data
        self.load_data()
        self.save_data()
        # self.test()

    def get_data_label(self, data_point):
        label = data_point.split(".")[-3]
        val = self.categories.index(label)
        return val

    def load_data(self):
        #i = 0
        for data_point in tqdm(os.listdir(self.DATA_DIR), desc=self.description):
            label = self.get_data_label(data_point)
            path_to_data_point = self.DATA_DIR + data_point
            resized_img_array = cv2.resize(cv2.imread(path_to_data_point, cv2.IMREAD_GRAYSCALE), (self.IMG_SIZE, self.IMG_SIZE))
            self.processed_data.append([resized_img_array, label])
            #i = i + 1
            #if i == 10: break
        shuffle(self.processed_data)

    def save_data(self):
        X = []
        y = []

        for feature, label in self.processed_data:
            X.append(feature)
            y.append(label)

        print("Saving data...")

        # reshaping data
        X = np.array(X).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        y = np.array(y)

        pickle_out = open("X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()
        pickle_out = open("y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

        print("Data saved successfully...")

    def test(self):
        pickle_in = open("X.pickle", "rb")
        X = pickle.load(pickle_in)
        pickle_in = open("y.pickle", "rb")
        y = pickle.load(pickle_in)
        print(X, y)
