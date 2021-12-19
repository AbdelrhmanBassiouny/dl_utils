import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


def optimizer_init_fn(optimizer_func=tf.keras.optimizers.Adam, lr=1e-3):
    return optimizer_func(learning_rate=lr)

def update_model_weights_from_ckpt(model_to_update, prev_run_path):
    prev_checkpoint_path = prev_run_path + "/training_2/cp-{epoch:04d}.ckpt"
    prev_checkpoint_dir = os.path.dirname(prev_checkpoint_path)
    
    latest = tf.train.latest_checkpoint(prev_checkpoint_dir)
    model_to_update.load_weights(latest)

def resize_and_save(X, y, size, base_path, batch_size=1024, file_type='png', scale=True, start=0, prefix='img'):
    x_mini_batches = [X[i: i + batch_size] for i in range(start, len(X), batch_size)]
    for i, x in enumerate(x_mini_batches):
        x = tf.keras.layers.experimental.preprocessing.Resizing(*size)(x)
        for j in range(0, x.shape[0]):
            img_idx = start + i * batch_size + j
            label = str(y[img_idx])
            img_dir = base_path + label + '/'
            os.makedirs(name=img_dir, exist_ok=True)
            path = img_dir + prefix + str(img_idx) + '.' + file_type
            tf.keras.preprocessing.image.save_img(path, x[j], data_format="channels_last", file_format=file_type, scale=scale)

def generate_data_from_dir(directory, generator=None, size=(224, 224), batch_size=64):
    if generator is None:
        genrator = ImageDataGenerator()
    return genrator.flow_from_directory(directory, target_size=size, class_mode='sparse', batch_size=batch_size)

# Put data in a tf.data.Dataset, preprocess and augment.
def prepare_dataset(data=None, generator=None, args=None, output_shape=(224, 224, 3), batch_size=64,
                    augmenter=None, preprocessor=None, shuffle=False, buffered_prefetching=False):
    
    if generator is not None:
        ds = tf.data.Dataset.from_generator(
            generator,args=args, output_types=(tf.float32, tf.int32), output_shapes=([None, *output_shape], [None]))
    else:
        ds = tf.data.Dataset.from_tensor_slices(data).batch(batch_size=batch_size)

    # Augment only train_ds
    if augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Resize and normalize all datasets
    if preprocessor is not None:
        ds = ds.map(lambda x, y: (preprocessor(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Shuffle dataset
    if shuffle:
        ds = ds.shuffle(buffer_size=10)

    # Use buffered prefecting on all datasets
    if buffered_prefetching:
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds
