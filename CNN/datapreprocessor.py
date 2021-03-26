from tqdm import tqdm
import time
import os
import numpy as np
import cv2
from random import shuffle

class DataPreProcessor:
    DATA_DIR = None
    IMG_SIZE = None
    DATA_SAVE_PATH = None
    processed_data = []
    description = None

    def __init__(self, options):
        self.DATA_DIR = options['DATA_DIR']
        self.IMG_SIZE = options['IMG_SIZE']
        self.DATA_SAVE_PATH = options['DATA_SAVE_PATH']
        self.description = options['DESCRIPTION']

    def start(self):
        # load training data
        self.load_data()
        self.save_data()

    def get_data_label(self, data_point):
        label = data_point.split(".")[-3]
        if label == 'dog':
            return [1, 0]
        else:
            return [0, 1]

    def load_data(self):
        for data_point in tqdm(os.listdir(self.DATA_DIR), desc=self.description):
            label = self.get_data_label(data_point)
            path_to_data_point = self.DATA_DIR + data_point
            #print(path_to_data_point)
            resized_img_array = cv2.resize(cv2.imread(path_to_data_point, cv2.IMREAD_GRAYSCALE), (self.IMG_SIZE, self.IMG_SIZE))
            self.processed_data.append([np.array(resized_img_array), np.array(label)])

        shuffle(self.processed_data)


    def save_data(self):
        print("Saving data...")
        np.save(self.DATA_SAVE_PATH, self.processed_data)