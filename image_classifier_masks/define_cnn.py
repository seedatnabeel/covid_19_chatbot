import importlib

from keras import backend as K
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

class conv_net:
    def __init__(self, pre_trained_model, n_classes=3, dropout=0, weights="imagenet"):
        assert pre_trained_model in (
            "InceptionResNetV2",
            "MobileNet",
            "VGG16",
        ), "Base Network Not Supported"
        self.baseModel = None
        self.headModel = None
        self.model = None
        self.pretrained_model = pre_trained_model
        self.image_shape = None
        self.n_classes = n_classes
        self.dropout = dropout
        self.weights = weights  # None or use imagenet weights
        self.load_pretrained()

    def load_pretrained(self):
        """ Loads a pre-trained conv net"""
        if self.pretrained_model == "InceptionResNetV2":
            self.model = importlib.import_module(
                "keras.applications.inception_resnet_v2"
            )
            IMAGE_WIDTH = 299
            IMAGE_HEIGHT = 299
        elif self.pretrained_model == "MobileNet":
            self.model = importlib.import_module("keras.applications.mobilenet")
            IMAGE_WIDTH = 224
            IMAGE_HEIGHT = 224
        elif self.pretrained_model == "VGG16":
            self.model = importlib.import_module("keras.applications.vgg16")
            IMAGE_WIDTH = 224
            IMAGE_HEIGHT = 224

        self.image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    def define_base_network(self):
        """Defines the base network or model as the pretrained network"""
        pre_trained = getattr(self.model, self.pretrained_model)

        self.baseModel = pre_trained(
            weights=self.weights, include_top=False, input_shape=self.image_shape
        )

        return self.baseModel

    def define_head_network(self):
        """Defines the head model (dense layers)"""
        headModel = self.baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = Dropout(self.dropout)(headModel)
        self.headModel = Dense(self.n_classes, activation="softmax")(headModel)

    def get_cnn(self):
        """ Creates the CNN, adding the head network to the base network"""
        self.custom_model = Model(inputs=self.baseModel.input, outputs=self.headModel)
        return self.custom_model
