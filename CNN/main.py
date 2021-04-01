# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from datapreprocessor import DataPreProcessor
from classifier1 import Classifier1

if __name__ == '__main__':
    options = {
        'DATA_DIR': 'H:\\code\\ML_Research_internship\\DataSets\\Dogs_Vs_Cats\\train\\',
        'IMG_SIZE': 50,
        'DATA_SAVE_PATH': 'H:\\code\\ML_Research_internship\\DataSets\\Dogs_Vs_Cats\\train_data.npy',
        'DESCRIPTION': 'Preprocessing training data',
        'CATEGORIES': ["dog", "cat"]
    }

    datapreprocessor = DataPreProcessor(options)
    classifier = Classifier1()
    #datapreprocessor.start()

    classifier.start()



