#-*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
from keras import layers as ly
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Concatenate
from keras import models as md#import md.Model
from keras import optimizers as op


class model_MLPunet(object):
    """
    img_rows,img_cols:设置图片大小
    num_md: 用于训练的模态个数
    loss_type: 使用哪种损失函数，1：binary_loss，2：使用dice_loss
    """
    def __init__(self, img_rows = 224, img_cols =224,num_md=1,loss_type=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_md = num_md
        self.loss_type = loss_type  #1:binary_cross, 2:dice_loss

    def ConvTokenizer(self,input, input_channels=None, output_channels=None, name='ConvTokenizer'):
        convtokenizer = ly.Conv2D(filters=input_channels//2, kernel_size=(3, 3),strides=2, padding=1)(input)
        convtokenizer = ly.BatchNormalization()(convtokenizer)
        convtokenizer = ly.Activation("Relu")(convtokenizer)
        convtokenizer = ly.Conv2D(filters=input_channels // 2, kernel_size=(3, 3), strides=2, padding=1)(convtokenizer)
        convtokenizer = ly.BatchNormalization()(convtokenizer)
        convtokenizer = ly.Activation("Relu")(convtokenizer)
        convtokenizer = ly.Conv2D(filters=input_channels, kernel_size=(3, 3), strides=2, padding=1)(convtokenizer)
        convtokenizer = ly.BatchNormalization()(convtokenizer)
        convtokenizer = ly.Activation("Relu")(convtokenizer)
        output = ly.MaxPooling2D(strides=2,kernel_size=(3, 3),padding=1,dilation=1)(convtokenizer)

        return output

    def ConvMLPBlock(self,input, input_channels=None, output_channels=None, name='MLP'):
        """
        ConvMLP block
        """
        if input_channels is None:
            input_channels = input.get_shape()[-1].value
        if output_channels is None:
            output_channels = input_channels

        layer_input = ly.LayerNormalization()(input)
        mlpblock1 = ly.Conv2D(filters=input_channels, kernel_size= (1, 1), strides=1)(layer_input)
        mlpblock1 = ly.Activation("GELU")(mlpblock1)
        mlpblock1 = ly.Conv2D(filters=output_channels, kernel_size=(1, 1), strides=1)(mlpblock1)

        layer_ln = ly.LayerNormalization()(mlpblock1)
        layer_conv = ly.Conv2D(filters=input_channels, kernel_size=(3, 3), strides=1,padding=1)(layer_ln)
        layer_ln = ly.LayerNormalization()(layer_conv)

        mlpblock2 = ly.Conv2D(filters=input_channels, kernel_size= (1, 1), strides=1)(layer_ln)
        mlpblock2 = ly.Activation("GELU")(mlpblock2)
        mlpblock2 = ly.Conv2D(filters=output_channels, kernel_size=(1, 1), strides=1)(mlpblock2)

        output = ly.add([input, mlpblock2])
        output = ly.Dropout(0.3)(output)

        return output

    def residual_block_2d(self,input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1, name='out'):
        """
        full pre-activation residual block
        https://arxiv.org/pdf/1603.05027.pdf
        """
        if output_channels is None:
            output_channels = input.get_shape()[-1].value
        if input_channels is None:
            input_channels = output_channels // 4

        strides = (stride, stride)

        x = ly.BatchNormalization()(input)
        x = ly.Activation('relu')(x)
        x = ly.Conv2D(input_channels, (1, 1))(x)

        x = ly.BatchNormalization()(x)
        x = ly.Activation('relu')(x)
        x = ly.Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

        x = ly.BatchNormalization()(x)
        x = ly.Activation('relu')(x)
        x = ly.Conv2D(output_channels, (1, 1), padding='same')(x)

        if input_channels != output_channels or stride != 1:
            input = ly.Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)
        if name == 'out':
            x = ly.add([x, input])
        else:
            x = ly.add([x, input], name=name)
        return x

    def build_res_atten_unet_2d(self, filter_num=8):
        merge_axis = -1  # Feature maps are concatenated along last axis (for tf backend)
        data = ly.Input((self.img_rows, self.img_cols, self.num_md))
        print("data", data.shape)

        conv1 = ly.Conv2D(filter_num * 4, 3, padding='same')(data)
        conv1 = ly.BatchNormalization()(conv1)
        conv1 = ly.Activation('relu')(conv1)

        pool = ly.MaxPooling2D(pool_size=(2, 2))(conv1)

        res1 = self.residual_block_2d(pool, output_channels=filter_num * 4)

        pool1 = ly.MaxPooling2D(pool_size=(2, 2))(res1)

        res2 = self.residual_block_2d(pool1, output_channels=filter_num * 8)

        pool2 = ly.MaxPooling2D(pool_size=(2, 2))(res2)

        res3 = self.residual_block_2d(pool2, output_channels=filter_num * 16)
        pool3 = ly.MaxPooling2D(pool_size=(2, 2))(res3)

        res4 = self.residual_block_2d(pool3, output_channels=filter_num * 32)

        pool4 = ly.MaxPooling2D(pool_size=(2, 2))(res4)

        res5 = self.residual_block_2d(pool4, output_channels=filter_num * 64)
        res5 = self.residual_block_2d(res5, output_channels=filter_num * 64)

        contokern = self.ConvTokenizer(res4)
        cmlpblock5 = self.ConvMLPBlock(contokern, name='mlp1')
        print("cmlpblock5", cmlpblock5)
        print("res5", res5)
        up1 = ly.UpSampling2D(size=(2, 2))(res5)
        merged1 = ly.concatenate([up1, cmlpblock5], axis=merge_axis)

        res6 = self.residual_block_2d(merged1, output_channels=filter_num * 32)

        contokern = self.ConvTokenizer(res3)
        cmlpblock6 = self.ConvMLPBlock(contokern, name='mlp2')
        print("cmlpblock6", cmlpblock6)
        up2 = ly.UpSampling2D(size=(2, 2))(res6)
        merged2 = ly.concatenate([up2, cmlpblock6], axis=merge_axis)

        res7 = self.residual_block_2d(merged2, output_channels=filter_num * 16)

        contokern = self.ConvTokenizer(res2)
        cmlpblock7 = self.ConvMLPBlock(contokern, name='mlp3')
        up3 = ly.UpSampling2D(size=(2, 2))(res7)
        merged3 = ly.concatenate([up3, cmlpblock7], axis=merge_axis)

        res8 = self.residual_block_2d(merged3, output_channels=filter_num * 8)

        contokern = self.ConvTokenizer(res1)
        cmlpblock8 = self.ConvMLPBlock(contokern, name='mlp4')
        up4 = ly.UpSampling2D(size=(2, 2))(res8)
        merged4 = ly.concatenate([up4, cmlpblock8], axis=merge_axis)

        res9 = self.residual_block_2d(merged4, output_channels=filter_num * 4)

        up = ly.UpSampling2D(size=(2, 2))(res9)
        merged = ly.concatenate([up, conv1], axis=merge_axis)

        conv9 = ly.Conv2D(filter_num * 4, 3, padding='same')(merged)
        conv9 = ly.BatchNormalization()(conv9)
        conv9 = ly.Activation('relu')(conv9)

        output = ly.Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)
        model = md.Model(data, output)

        model.compile(optimizer=op.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      loss="binary_crossentropy",
                      metrics=[dice_coef_eval])
        return model

def binary_crossentropy(y_true,y_pred):
    return tl.cost.binary_cross_entropy(y_pred[0],y_true[0]) + tl.cost.binary_cross_entropy(y_pred[1],y_true[0])

def dice_coef_eval(y_true, y_pred):  # dice距离：DS=(2*(‖X‖∩‖Y‖)/(‖X‖+‖Y‖))
     return tl.cost.dice_coe(y_pred,y_true,loss_type='jaccard', axis=[1,2,3],smooth=0.01)

def dice_loss_2d(Y_gt, Y_pred):
    H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = -tf.reduce_mean(intersection / denominator)
    return loss


def dice_coef_loss(y_true, y_pred):
    return 1 - tl.cost.dice_coe(y_pred,y_true,loss_type='jaccard', axis=[1,2,3],smooth=0.5)

def SoftmaxWithLoss(y_true, y_pred):
    print(y_pred)
    print(y_true)
    loss = tf.nn.softmax_cross_entropy_with_logits(y_true,y_pred)
    return loss
    
    
def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)
        
