import unittest
from image_classifier_masks.define_cnn import *

import importlib

from keras import backend as K
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

class Test_Build_Image_Classifier(unittest.TestCase):

    def test_get_cnn(self):
        cnn = conv_net(
            pre_trained_model="MobileNet",
            n_classes=3,
            dropout=0,
            weights='imagenet',
        )

        # Define your base pre-trained network
        baseModel = cnn.define_base_network()
        # Define the head network (fully connected layers for the task)
        cnn.define_head_network()

        # get the CNN model
        cnn_model = cnn.get_cnn()

        self.assertIsInstance(cnn_model, Model)

    def test_base_model(self):
        cnn = conv_net(
            pre_trained_model="MobileNet",
            n_classes=3,
            dropout=0,
            weights='imagenet',
        )
        # Define your base pre-trained network
        baseModel = cnn.define_base_network()

        self.assertIsInstance(baseModel, Model)



if __name__ == '__main__':
    unittest.main()
