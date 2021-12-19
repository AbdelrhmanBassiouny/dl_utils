from tensorflow_utils.classifiers.tf_nets import MiniInception, MiniAlexNet, ResNet, DenseNet
from tensorflow_utils.classifiers.tf_func_nets import mini_alex_net, mini_inception, res_net, dense_net
from tensorflow_utils.classifiers.tf_layers import ConvNormalizeDrop
from tensorflow_utils.utils.keras_load_data import load_cifar10, Dataset, get_generator_for
import tensorflow as tf
import time
import os
from tensorflow_utils.utils.tf_utils import *


def model_init_fn():
    global nets, net_name
    net = nets[net_name][0]
    inputs = nets[net_name][1]
    return net(**inputs)

if __name__ == "__main__":

    from_generator = False
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(augment=not from_generator,
                                                                  normalize=not from_generator)

    if from_generator:
        data_gen, (X_val, y_val), (X_test, y_test) = get_generator_for((X_train, y_train), (X_val, y_val),
                                                                       (X_test, y_test),
                                                                       augment=True)

    device = '/device:GPU:0'  # Change this to a CPU/GPU as you wish!
    #     device = '/cpu:0'        # Change this to a CPU/GPU as you wish!
    num_epochs = 20 if from_generator else 10
    bs = 64
    learning_rate = 1e-3
    lr_decay = 0.5
    decay_every = 'val_acc_plateau'
    d, l1, l2 = 0.03, 0, 0
    reg = {'l1': l1, 'l2': l2, 'drop_prop': d}

    nets = {
        'MiniInception': (MiniInception, reg),
        'NaiveMiniInception': (MiniInception, {'params': (64, 32, 16), 'naive': True, **reg}),
        'MiniAlexAffine': (MiniAlexNet, {'affine_size': [512] * 2, **reg}),
        'MiniAlexGavg': (MiniAlexNet, {'use_avg_pool': True, **reg}),
        'NaiveResNet3_2': (ResNet, {'reduce_every': 3, 'repeats': 2, **reg}),  # 9 blocks = 18 learnable layers
        'NaiveResNet2_3': (ResNet, {'reduce_every': 2, 'repeats': 3, **reg}),  # 8 blocks = 16 learnable layers
        'NaiveResNet2_2': (ResNet, {'reduce_every': 2, 'repeats': 2, **reg}),  # 6 blocks = 12 learnable layers
        'NaiveResNet2_1': (ResNet, {'reduce_every': 2, 'repeats': 1, **reg}),  # 4 blocks = 8 learnable layers
        'dense_net_6_12_32_k12': (DenseNet,
                                  {'dense_layers': (6, 12, 32), 'k': 12, 'compactness': 0.5, 'pool': 'avg', **reg})
    }

    net_name = 'dense_net_6_12_32_k12'

    model_name = '{}_l1={}_l2={}_drop={}'.format(net_name, l1, l2, d)

    dir_name = "/home/abdelrhman/loggings/"

    run_name = dir_name + model_name + \
        '_lr{}_decay{}_every{}_'.format(learning_rate, lr_decay, decay_every) + \
        'hflip_zpad_crop' + time.strftime("run_%Y_%m_%d-%H_%M_%S")

    dot_img_file = run_name + '/model_img.png'

    print(run_name)

    # Define the Keras TensorBoard callback.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_name)
    lr_on_plateau_decay = tf.keras.callbacks.ReduceLROnPlateau('val_sparse_categorical_accuracy',
                                                               lr_decay, patience=1, verbose=1)

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = run_name + "/training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=2)

    with tf.device(device):

        # Use this when u need to use the subclass form of the model.
        #     model = model_init_fn()

        # Use this when u need to use the functional form of the model (can visualize, summarize and optimize ^_^)
        model = model_init_fn().get_functional_model(X_train.shape)

        image_batch, label_batch = next(data_gen)
        base_model = load_pre_trained_model()
        feature_batch = base_model(image_batch)
        print(feature_batch.shape)

        print(model.summary())

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=[tf.keras.metrics.sparse_categorical_accuracy])

        if from_generator:
            # fits the model on batches with real-time data augmentation:
            model.fit(data_gen.flow(X_train, y_train, batch_size=bs),
                      steps_per_epoch=len(X_train) / bs, epochs=num_epochs,
                      validation_data=(X_val, y_val), verbose=2, callbacks=[tensorboard_callback, cp_callback])
        else:
            model.fit(X_train, y_train, batch_size=bs, epochs=num_epochs,
                      validation_data=(X_val, y_val), verbose=2, callbacks=[tensorboard_callback, cp_callback])
        model.save(run_name + '/model')

        model.evaluate(X_test, y_test, batch_size=64, callbacks=[tensorboard_callback])
# ds = tf.data.Dataset.from_generator(
#     train_iter, args=[flowers],
#     output_types=(tf.float32, tf.float32),
#     output_shapes=([32,256,256,3], [32,5])
# )
