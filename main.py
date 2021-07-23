#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Concatenate, Conv2D, Conv2DTranspose, Cropping2D, Input, MaxPooling2D, SeparableConv2D

# https://idiotdeveloper.com/unet-implementation-in-tensorflow-using-keras-api/

def conv_block(input_, num_filters):
	x = SeparableConv2D(num_filters, 3, activation="relu")(input_)
	x = SeparableConv2D(num_filters, 3, activation="relu")(x)

	return x

def encoder_block(input_, num_filters):
	x = conv_block(input_, num_filters)

	return MaxPooling2D()(x), x

def cropping_margins(src_shape, target_shape):
	# https://github.com/karolzak/keras-unet/blob/9b7aff5247fff75dc4e2a11ba9c45929b9166d1f/keras_unet/models/vanilla_unet.py

	delta_h = src_shape[1] - target_shape[1]

	top_crop = delta_h // 2
	bottom_crop = delta_h // 2 if delta_h % 2 == 0 else delta_h // 2 + 1

	delta_w = src_shape[2] - target_shape[2]

	left_crop = delta_w // 2
	right_crop = delta_w // 2 if delta_w % 2 == 0 else delta_w // 2 + 1

	return ((top_crop, bottom_crop), (left_crop, right_crop))


def decoder_block(input_, skipped_features, num_filters):
	x = Conv2DTranspose(num_filters, (2, 2), strides=2)(input_)
	cropped = Cropping2D(cropping=cropping_margins(int_shape(skipped_features), int_shape(x)))(skipped_features)
	x = Concatenate()([x, cropped])
	x = conv_block(x, num_filters)

	return x

def get_model(input_shape, output_channels):
	inputs = Input(input_shape)

	assert input_shape[0] >= 140 and (input_shape[0] - 124) % 16 == 0, "Input width must be of the form 16x + 124"
	assert input_shape[1] >= 140 and (input_shape[1] - 124) % 16 == 0, "Input height must be of the form 16x + 124"

	e1, s1 = encoder_block(inputs, 64)
	e2, s2 = encoder_block(e1, 128)
	e3, s3 = encoder_block(e2, 256)
	e4, s4 = encoder_block(e3, 512)

	bridge = conv_block(e4, 1024)

	d1 = decoder_block(bridge, s4, 512)
	d2 = decoder_block(d1, s3, 256)
	d3 = decoder_block(d2, s2, 128)
	d4 = decoder_block(d3, s1, 64)

	outputs = Conv2D(output_channels, 1, activation="sigmoid")(d4)

	return Model(inputs=inputs, outputs=outputs)

# https://www.tensorflow.org/tutorials/images/segmentation

dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)

# https://www.tensorflow.org/guide/data_performance

# At the bottom of the U-Net (after the last convolution that bridges the encoder and decoder), the
# image has the most features and the lowest resolution.
MINIMUM_SHAPE = (12, 12)

INPUT_SHAPE = (MINIMUM_SHAPE[0] * 16 + 124, MINIMUM_SHAPE[1] * 16 + 124)
OUTPUT_SHAPE = (MINIMUM_SHAPE[0] * 16 - 60, MINIMUM_SHAPE[1] * 16 - 60)

def load_datapoint(datapoint, random_flip=False):
	input_image = tf.image.resize(datapoint["image"], OUTPUT_SHAPE)
	input_image = tf.cast(input_image, tf.float32) / 255
	input_image = tf.pad(input_image, ((92, 92), (92, 92), (0, 0)), mode="SYMMETRIC")
	input_mask = tf.image.resize(datapoint["segmentation_mask"], OUTPUT_SHAPE)

	if random_flip and tf.random.uniform(()) > 0.5:
		input_image = tf.image.flip_left_right(input_image)
		input_mask = tf.image.flip_left_right(input_mask)

	# The dataset's labels are within {1, 2, 3}, but the sparse categorical crossentropy loss
	# function expects them to be within [0, 3)
	input_mask -= 1

	return input_image, input_mask

BATCH_SIZE = 6

train_dataset = (dataset["train"]
	.map(lambda d: load_datapoint(d, True), num_parallel_calls=tf.data.AUTOTUNE)
	.cache()
	.shuffle(16) # https://datascience.stackexchange.com/a/89319
	.batch(BATCH_SIZE)
	.repeat() # https://www.gcptutorials.com/article/how-to-use-tf.data.Dataset.repeat
	.prefetch(buffer_size=tf.data.AUTOTUNE))

test_dataset = (dataset["test"]
	.map(load_datapoint)
	.batch(BATCH_SIZE))

model = get_model((INPUT_SHAPE[0], INPUT_SHAPE[1], 3), 3)
model.compile(
	optimizer="adam",
	loss="sparse_categorical_crossentropy", # https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c
	metrics="accuracy")

model.summary()

CHECKPOINT_PATH = "weights/cp.ckpt"

# https://www.tensorflow.org/tutorials/keras/save_and_load
try:
	model.load_weights(CHECKPOINT_PATH)
except:
	print("No weights were loaded.")

model.fit(
	train_dataset,
	epochs=20,
	callbacks=[tf.keras.callbacks.ModelCheckpoint(
		filepath="weights/cp.ckpt",
		save_weights_only=True,
    	verbose=1)],
	validation_data=test_dataset,
	steps_per_epoch=info.splits['train'].num_examples // BATCH_SIZE,
	validation_steps=info.splits["test"].num_examples)
