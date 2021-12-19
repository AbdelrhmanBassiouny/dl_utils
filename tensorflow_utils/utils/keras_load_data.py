import tensorflow as tf
import numpy as np
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt


def plot_sample_images(images, labels, num_classes, samples_per_class, mean=None, std=None, dtype='uint8'):
    print(images[0].shape)
    if mean is not None and std is not None:
        images = images * std + mean
    for y in range(num_classes):
        idxs = np.flatnonzero(labels == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(images[idx, :, :, :].astype(dtype))
            plt.axis('off')


def load_cifar10(num_training=49000, num_validation=1000, num_test=10000, augment=False,
                 rot=0, pad_px=4, h_flip_prop=0.5, size=(32, 32), normalize=True, plot=True):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    if augment:
    
        # Augment the train data
        augmenter = iaa.Sequential([iaa.Fliplr(h_flip_prop),
                                    iaa.Rotate((-rot, rot)),
                                    # iaa.Flipud(0.5),
                                    # iaa.Crop(px=(4, 10), keep_size=False),
                                    # iaa.PadToFixedSize(32, 32),
                                    iaa.Pad(px=pad_px, keep_size=False),
                                    iaa.CropToFixedSize(width=32, height=32),
                                    iaa.Resize(size)
                                    # iaa.RandAugment()
                                    ])

        X_train_list = augmenter.augment_images(X_train.astype('uint8'))
        
        X_train_augmented = np.array(X_train_list, dtype='float32').reshape((X_train.shape[0], *size, X_train.shape[3]))
                
        X_train = np.vstack([X_train_augmented, X_train])
        y_train = np.hstack([y_train, y_train])
        # X_train = X_train_augmented

        # np.random.seed(1)
        idxs = list(range(0, X_train.shape[0]))
        np.random.shuffle(idxs)

        X_train[:, :, :, :] = X_train[idxs, :, :, :]
        y_train[:] = y_train[idxs]

    if plot:
        plot_sample_images(X_train, y_train, 10, 7)

    if normalize:
        # Normalize the data: subtract the mean pixel and divide by std
        mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
        std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
        X_train = (X_train - mean_pixel) / std_pixel
        X_val = (X_val - mean_pixel) / std_pixel
        X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_generator_for(train_data=None, augment=False,
                      rot=0, w_shift=0.125, h_shift=0.125, h_flip=False,
                      normalize=True, plot=True):

    if train_data:
        X_train, y_train = train_data

    processes = {'featurewise_center': normalize, 'featurewise_std_normalization': normalize}

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**processes)

    if augment:
        processes.update({'rotation_range': rot,
                          'width_shift_range': w_shift,
                          'height_shift_range': h_shift,
                          'horizontal_flip': h_flip,
                          'fill_mode': 'constant'})

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(**processes)

    if plot and train_data:
        images, labels = next(datagen.flow(X_train, y_train, batch_size=len(X_train) // 2))
        plot_sample_images(images, labels, 10, 7)

    if normalize and train_data:
        datagen.fit(X_train)
        val_datagen.fit(X_train)

    return datagen, val_datagen


class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle
        self.augment = iaa.Sequential([iaa.Fliplr(0.5),
                                       # iaa.Flipud(0.5),
                                       # iaa.Crop(px=(4, 10), keep_size=False),
                                       # iaa.PadToFixedSize(32, 32),
                                       iaa.Pad(px=4, keep_size=False),
                                       iaa.CropToFixedSize(width=32, height=32),
                                       # iaa.RandAugment()
                                       ])

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.augment(images=self.X[i:i + B]), self.y[i:i + B]) for i in range(0, N, B))
