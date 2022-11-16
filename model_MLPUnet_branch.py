import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Resizing, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from keras import layers
from tensorflow.keras.layers import Embedding, Input, LayerNormalization, Dense, GlobalAveragePooling1D, Dropout,BatchNormalization,Activation,Conv2D,Add,MaxPooling2D,UpSampling2D,concatenate,LayerNormalization


class Patches(layers.Layer):
  def __init__(self, patch_size):
    super(Patches, self).__init__()
    self.patch_size = patch_size

  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images= images,
        sizes = [1, self.patch_size, self.patch_size, 1],
        strides = [1, self.patch_size, self.patch_size, 1],
        rates = [1, 1, 1, 1],
        padding = 'VALID',
    )
    dim = patches.shape[-1]

    patches = tf.reshape(patches, (batch_size, -1, dim))
    return patches

class MLPBlock(tf.keras.layers.Layer):
  def __init__(self, S, C, DS, DC):
    super(MLPBlock, self).__init__()
    self.layerNorm1 = LayerNormalization()
    self.layerNorm2 = LayerNormalization()
    w_init = tf.random_normal_initializer()
    self.DS = DS
    self.DC = DC
    self.W1 = tf.Variable(
            initial_value=w_init(shape=(S, DS), dtype="float32"),
            trainable=True,
    )
    self.W2 = tf.Variable(
            initial_value=w_init(shape=(DS, S), dtype="float32"),
            trainable=True,
    )
    self.W3 = tf.Variable(
            initial_value=w_init(shape=(C, DC), dtype="float32"),
            trainable=True,
    )
    self.W4 = tf.Variable(
            initial_value=w_init(shape=(DC, C), dtype="float32"),
            trainable=True,
    )

  def call(self, X):
    # patches (..., S, C)
    batch_size, S, C = X.shape

    # Token-mixing
    # (..., C, S)
    X_T = tf.transpose(self.layerNorm1(X), perm=(0, 2, 1))

    # assert X_T.shape == (batch_size, C, S), 'X_T.shape: {}'.format(X_T.shape)

    W1X = tf.matmul(X_T, self.W1) # (..., C, S) . (S, DS) = (..., C, DS)

    # (..., C, DS) . (DS, S) == (..., C, S)
    # (..., C, S). T == (..., S, C)
    # (..., S, C) + (..., S, C) = (..., S, C)
    U = tf.transpose(tf.matmul(tf.nn.gelu(W1X), self.W2), perm=(0, 2, 1)) + X

    # Channel-minxing

    W3U = tf.matmul(self.layerNorm2(U), self.W3) # (...,S, C) . (C, DC) = (..., S, DC)

    Y = tf.matmul(tf.nn.gelu(W3U), self.W4) + U  # (..., S, DC) . (..., DC, C) + (..., S, C) = (..., S, C)

    return Y

class MLPMixerUnet(tf.keras.models.Model):
  def __init__(self, patch_size, S, C, num_of_mlp_blocks, image_size, batch_size):
    super(MLPMixerUnet, self).__init__()

    self.batch_size = batch_size
    self.patch_size = patch_size
    self.S = S
    self.C = C
    self.num_of_mlp_blocks = num_of_mlp_blocks
    self.image_size = image_size

    self.data_augmentation = tf.keras.Sequential(
        [
            Normalization(),
            # Resizing(image_size, image_size),
            RandomFlip("horizontal"),
            RandomRotation(factor=0.02),
            RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )

    self.outputLayer = Sequential([
          GlobalAveragePooling1D(),
          Dropout(0.2),
    ])

  def extract_patches(self, images, patch_size):
    batch_size = tf.shape(images)[0]

    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    patches = tf.reshape(patches, [batch_size, -1, 3 * patch_size ** 2])

    return patches

  def residual_block_2d(self,input, input_channels = None, output_channels=None, kernel_size=(3, 3), stride=1, name='out'):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape()[-1].value
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)
    if name == 'out':
        x = Add([x, input])
    else:
        x = Add([x, input], name=name)
    return x

  def residual_unet(self,images):
      # input
      # images shape: (batch_size, image_size, image_size, 3) = (32, 64, 64, 3)
      filter_num = 8
      merge_axis = -1
      batch_size = images.shape[0]

      augumented_images = self.data_augmentation(images)

      # assert augumented_images.shape == (batch_size, self.image_size, self.image_size, 3)

      # patches shape: (batch_size, S, 3 * patch_size ** 2)
      X = self.extract_patches(augumented_images, self.patch_size)

      conv1 = Conv2D(self.filter_num * 4, 3, padding='same')(X)
      conv1 = BatchNormalization()(conv1)
      conv1 = Activation('relu')(conv1)
      pool = MaxPooling2D(pool_size=(2, 2))(conv1)

      res1 = self.residual_block_2d(pool, output_channels=filter_num * 4)
      pool1 = MaxPooling2D(pool_size=(2, 2))(res1)

      res2 = self.residual_block_2d(pool1, output_channels=filter_num * 8)
      pool2 = MaxPooling2D(pool_size=(2, 2))(res2)

      res3 = self.residual_block_2d(pool2, output_channels=filter_num * 16)
      pool3 = MaxPooling2D(pool_size=(2, 2))(res3)

      res4 = self.residual_block_2d(pool3, output_channels=filter_num * 32)
      pool4 = MaxPooling2D(pool_size=(2, 2))(res4)

      res5 = self.residual_block_2d(pool4, output_channels=filter_num * 64)
      res5 = self.residual_block_2d(res5, output_channels=filter_num * 64)

      up1 = UpSampling2D(size=(2, 2))(res5)
      merged1 = concatenate([up1, res4], axis=merge_axis)

      res5 = self.residual_block_2d(merged1, output_channels=filter_num * 32)

      up2 = UpSampling2D(size=(2, 2))(res5)
      merged2 = concatenate([up2, res3], axis=merge_axis)

      res6 = self.residual_block_2d(merged2, output_channels=filter_num * 16)

      up3 = UpSampling2D(size=(2, 2))(res6)
      merged3 = concatenate([up3, res2], axis=merge_axis)

      res7 = self.residual_block_2d(merged3, output_channels=filter_num * 8)

      up4 = UpSampling2D(size=(2, 2))(res7)
      merged4 = concatenate([up4, res1], axis=merge_axis)

      res8 = self.residual_block_2d(merged4, output_channels=filter_num * 4)

      up = UpSampling2D(size=(2, 2))(res8)
      merged = concatenate([up, conv1], axis=merge_axis)

      conv9 = Conv2D(filter_num * 4, 3, padding='same')(merged)
      conv9 = BatchNormalization()(conv9)
      conv9 = Activation('relu')(conv9)

      return conv9

  def mlp_mixer_branch(self, images):
      # input
      # images shape: (batch_size, image_size, image_size, 3) = (32, 64, 64, 3)

      batch_size = images.shape[0]

      augumented_images = self.data_augmentation(images)

      # assert augumented_images.shape == (batch_size, self.image_size, self.image_size, 3)

      # patches shape: (batch_size, S, 3 * patch_size ** 2)
      X = self.extract_patches(augumented_images, self.patch_size)

      # Per-patch Fully-connected
      # X shape: (batch_size, S, C)
      X = self.projection(X)

      # assert X.shape == (batch_size, self.S, self.C)
      for block in self.mlpBlocks:
          X = block(X)

      # assert X.shape == (batch_size, self.S, self.C)

      # out shape: (batch_size, C)
      out = self.outputLayer(X)

      return out

  def call(self, images):
    #combine resunet and mlpmixerbranch
    mixer_layer = self.residual_unet(images)+self.mlp_mixer_branch(images)
    output = Conv2D(1, 1, padding='same', activation='sigmoid')(mixer_layer)

    # assert out.shape == (batch_size, self.num_classes)
    return output


