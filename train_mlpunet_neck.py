import os
from argparse import ArgumentParser
import tensorflow as tf
from model_MLPUnet_neck import MLPMixerUnet_neck
import logging
from keras import callbacks as cbk
logging.basicConfig(level=logging.DEBUG)

#设置阀值，
def dice_coef_eval(y_true, y_pred):  # dice距离：DS=(2*(‖X‖∩‖Y‖)/(‖X‖+‖Y‖))
     return tf.cost.dice_coe(y_pred,y_true,loss_type='jaccard', axis=[1,2,3],smooth=0.01)

def dice_coef_loss(y_true, y_pred):
    #y_pred[y_pred>=0.5] = 1
    #y_pred[y_pred<0.5] = 0
   # print(y_true)
    return 1 - tf.cost.dice_coe(y_pred,y_true,loss_type='jaccard', axis=[1,2,3],smooth=0.5)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()
    # parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--train-folder", default='{}/data/train'.format(home_dir), type=str)
    parser.add_argument("--valid-folder", default='{}/data/validation'.format(home_dir), type=str)
    parser.add_argument("--model-folder", default='{}/model/mlp/'.format(home_dir), type=str)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--dc", default=2048, type=int, help='Token-mixing units')
    parser.add_argument("--ds", default=256, type=int, help='Channel-mixing units')
    parser.add_argument("--c", default=512, type=int, help='Projection units')
    parser.add_argument("--image-size", default=150, type=int)
    parser.add_argument("--patch-size", default=5, type=int)
    parser.add_argument("--num-of-mlp-blocks", default=8, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--validation-split", default=0.2, type=float)
    parser.add_argument("--image-channels", default=3, type=int)
    parser.add_argument("--model-weight", default='{}/weight'.format(home_dir),type=str)

    args = parser.parse_args()

    print('---------------------Welcome to ProtonX MLP Mixer-------------------')
    print('Github: bangoc123')
    print('Email: protonxai@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training MLP-Mixer model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')



    train_folder = args.train_folder
    valid_folder = args.valid_folder

    # Load train images from folder
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_folder,
        subset="training",
        seed=123,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        validation_split = args.validation_split,
        batch_size=args.batch_size,
    )

    # Load Validation images from folder
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_folder,
        subset="validation",
        seed=123,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        validation_split = args.validation_split,
        batch_size= args.batch_size,
    )

    assert args.image_size * args.image_size % ( args.patch_size * args.patch_size) == 0, 'Make sure that image-size is divisible by patch-size'
    assert args.image_channels == 3, 'Unfortunately, model accepts jpg images with 3 channels so far'
    
    S = (args.image_size * args.image_size) // (args.patch_size * args.patch_size)
    # C = args.patch_size * args.patch_size * args.image_channels

    
    # Initializing model
    mlpmixerunet = MLPMixerUnet_neck(S, args.c, args.num_of_mlp_blocks, args.image_size, args.batch_size)

    # Set up loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Optimizer Definition
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Compile optimizer and loss function into model
    mlpmixerunet.compile(optimizer=adam,
                  loss="binary_crossentropy",
                  metrics=['Dice',dice_coef_eval])

    hdf5_file = "{0}/mlpunet_neck.hdf5".format(args.model_weight)
    model_checkpoint = cbk.ModelCheckpoint(hdf5_file, monitor='loss', verbose=1,
                                           save_weights_only=True,
                                           save_best_only=True, mode='min')
    reduce_lrate = cbk.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, mode='min', min_lr=0)
    early_stopiing = cbk.EarlyStopping(monitor='dice_coef_eval', patience=15, verbose=1, mode='max')

    print('Fitting md.Model...')
    # Do Training model
    historys = mlpmixerunet.fit(train_ds,
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        validation_data=val_ds,
        callbacks=[model_checkpoint, early_stopiing],
    )

    # Saving model
    mlpmixerunet.save(args.model_folder)

    loss_file = "{0}/training_loss_dc.txt".format(args.model_weight)
    with open(loss_file, 'w') as fp:
        for record in historys.history:
            line = "%s %s\n" % (record, ' '.join([str(val) for val in historys.history[record]]))
            fp.write(line)