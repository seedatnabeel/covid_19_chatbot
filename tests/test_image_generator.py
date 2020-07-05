import unittest
from image_classifier_masks.generator import *

from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


class Test_Image_Generator(unittest.TestCase):

    def test_load_ref_data(self):
        train_gen, val_gen = data_generator(
            data_dir='.',
            batch_size=32,
            validation_size=0.2,
        )
        self.assertIsNotNone(train_gen)
        self.assertIsNotNone(val_gen)

if __name__ == '__main__':
    unittest.main()
