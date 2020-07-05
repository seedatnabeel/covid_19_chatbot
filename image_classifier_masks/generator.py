from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


def data_generator(
    data_dir,
    batch_size,
    validation_size,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    class_mode="categorical",
    target_size=(224, 224),
):
    """
    Training and Validation data generator

    Args:

    data_dir: directory where image data is stored
    batch_size (int): training batch size
    validation_size (0-1): validation set size
    horizontal_flip (bool): image augmentation whether to do flip
    zoom_range (0-1): image augmentation range of zoom 0-1
    shear_range (0-1) : image augmentation range of shear 0-1
    class_mode (str) : 'binary' or 'categorical'
    target_size: target size of image based on the pre-trained network

    Returns:
        training_set, val_set
    """

    # Define train and test generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=shear_range,
        zoom_range=zoom_range,
        validation_split=validation_size,
        horizontal_flip=horizontal_flip,
    )

    # since it's validation/test we don't apply augmentations to the images
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    training_set = train_datagen.flow_from_directory(
        data_dir,
        target_size=(target_size[0], target_size[1]),
        batch_size=batch_size,
        subset="training",
        class_mode=class_mode,
    )

    val_set = test_datagen.flow_from_directory(
        data_dir,
        target_size=(target_size[0], target_size[1]),
        batch_size=batch_size,
        subset="validation",
        class_mode=class_mode,
    )

    return training_set, val_set
