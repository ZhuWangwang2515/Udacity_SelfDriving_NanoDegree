#!/usr/bin/env python3
import os
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import glob
import numpy as np
# import cv2
import sys                                                              
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')         
import cv2                                                              
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')    
import random     


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def perform_augmentation(batch_x, batch_y):
    """
    Perform basic data augmentation on image batches.
    
    Parameters
    ----------
    batch_x: ndarray of shape (b, h, w, c)
        Batch of images in RGB format, values in [0, 255]
    batch_y: ndarray of shape (b, h, w, c)
        Batch of ground truth with road segmentation
        
    Returns
    -------
    batch_x_aug, batch_y_aug: two ndarray of shape (b, h, w, c)
        Augmented batches
    """
    def mirror(x):
        return x[:, ::-1, :]

    def augment_in_hsv_space(x_hsv):
        x_hsv = np.float32(cv2.cvtColor(x_hsv, cv2.COLOR_RGB2HSV))
        x_hsv[:, :, 0] = x_hsv[:, :, 0] * random.uniform(0.9, 1.1)   # change hue
        x_hsv[:, :, 1] = x_hsv[:, :, 1] * random.uniform(0.5, 2.0)   # change saturation
        x_hsv[:, :, 2] = x_hsv[:, :, 2] * random.uniform(0.5, 2.0)   # change brightness
        x_hsv = np.uint8(np.clip(x_hsv, 0, 255))
        return cv2.cvtColor(x_hsv, cv2.COLOR_HSV2RGB)

    batch_x_aug = np.copy(batch_x)
    batch_y_aug = np.copy(batch_y)

    for b in range(batch_x_aug.shape[0]):

        # Random mirroring
        should_mirror = random.choice([True, False])
        if should_mirror:
            batch_x_aug[b] = mirror(batch_x[b])
            batch_y_aug[b] = mirror(batch_y[b])

        # Random change in image values (hue, saturation, brightness)
        batch_x_aug[b] = augment_in_hsv_space(batch_x_aug[b])

    return batch_x_aug, batch_y_aug



def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    W1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    Keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    W2 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    W3 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    W4 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return W1, Keep, W2, W3, W4


tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    layer3_logits = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=[1, 1], padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer4_logits = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=[1, 1], padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer7_logits = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=[1, 1], padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer7_logits_up = tf.image.resize_images(layer7_logits, size=[10, 36])
    layer_4_7_fused = tf.add(layer7_logits_up, layer4_logits)

    layer_4_7_fused_up = tf.image.resize_images(layer_4_7_fused, size=[20, 72])
    layer_3_4_7_fused = tf.add(layer3_logits, layer_4_7_fused_up)

    layer_3_4_7_up = tf.image.resize_images(layer_3_4_7_fused, size=[160, 576])
    layer_3_4_7_up = tf.layers.conv2d(layer_3_4_7_up, num_classes, kernel_size=[15, 15], padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return layer_3_4_7_up

tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    logits_flat = tf.reshape(nn_last_layer, (-1, num_classes))
    labels_flat = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_flat, logits=logits_flat))

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits_flat, train_step, cross_entropy_loss

tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    image_paths = glob.glob(os.path.join('data/data_road/training/image_2', '*.png'))
    learning_rate_ = 1e-4
    for i in range(epochs):
        loss_current_epoch = 0.0
        for offset in range(0, len(image_paths), batch_size):

            batch_x, batch_y = next(get_batches_fn(batch_size))
            # batch_x, batch_y = get_batches_fn(batch_size).__next__()

            _, cur_loss = sess.run(fetches=[train_op, cross_entropy_loss],
                                   feed_dict={input_image: batch_x, correct_label: batch_y, keep_prob: 0.25,
                                              learning_rate: learning_rate_})

            loss_current_epoch += cur_loss

        print('Epoch: {:02d}  -  Loss: {:.03f}'.format(i, loss_current_epoch / len(batch_x)))

tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    epochs = 30
    batch_size = 8
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        labels = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # print(1111111111111111111111111111111111111)
        # Create generator to get batches
        get_batches_generator = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        # print(1111111111111111111111111111111111111)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output, labels, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_generator, train_op, cross_entropy_loss,
                 image_input, labels, keep_prob, learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    
    # data_dir = './pretrained_model'
    # helper.maybe_download_pretrained_vgg(data_dir)
    run()