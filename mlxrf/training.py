

import numpy
from mlcrl.get_wofry_data import get_wofry_data
from tensorflow.keras import layers
from tensorflow.keras import models

from srxraylib.plot.gol import plot

from keras.models import load_model
import pickle

from keras.optimizers import RMSprop
import json

from srxraylib.plot.gol import plot_image

def get_model(
    architecture = "convnet", # not used!
    # kernel_size = (3, 3),
    # pool_size = (2, 2),
    kernel_size=(3, 1),
    pool_size=(2, 1),
    activation = 'relu', # 'tanh', #  'softmax'
    padding = 'same',
    # input_shape = tuple((256, 64, 1)),
    # output_size = 7,
    input_shape=tuple((8101, 1)),
    output_size=87,
    ):

    model = models.Sequential()

    model.add(layers.Conv2D(8, name='conv1', kernel_size=kernel_size, activation=activation, padding=padding,
                            input_shape=input_shape))
    model.add(layers.Conv2D(8, name='conv2', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling2D(name='maxpool1', pool_size=pool_size))

    model.add(layers.Conv2D(16, name='conv3', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv2D(16, name='conv4', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling2D(name='maxpool2', pool_size=pool_size))

    model.add(layers.Conv2D(32, name='conv5', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv2D(32, name='conv6', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling2D(name='maxpool3', pool_size=pool_size))

    model.add(layers.Conv2D(64, name='conv7', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv2D(64, name='conv8', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.MaxPooling2D(name='maxpool4', pool_size=pool_size))

    model.add(layers.Conv2D(128, name='conv9', kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(layers.Conv2D(128, name='conv10', kernel_size=kernel_size, activation=activation, padding=padding))
    try:
        if input_shape[0] == 1:
            model.add(layers.MaxPooling2D(name='maxpool5', pool_size=(1, 2)))
        else:
            model.add(layers.MaxPooling2D(name='maxpool5', pool_size=(2, 2)))
    except:
        model.add(layers.MaxPooling2D(name='maxpool5', pool_size=(1, 1)))

    model.add(layers.Flatten(name='flat'))
    model.add(layers.Dense(64, name='dense1', activation=activation))
    model.add(layers.Dense(64, name='dense2', activation=activation))
    model.add(layers.Dense(output_size, name='Y', activation='linear'))

    print(model.summary())
    return model

def load_training_set(nsamples=30, verbose=1, nbin=1, training_ratio=2/3, directory="./data1/"):
    a = numpy.loadtxt(directory + "sampled_00000_spe.dat")
    ashape = a.shape
    targets = numpy.zeros((nsamples, a.shape[0]))

    b = numpy.loadtxt(directory + "sampled_00000_xrf.dat")
    bshape = b.shape
    data = numpy.zeros((nsamples, b.shape[0], 1, 1))
    #                                         ^ fake, to use Conv2D
    #                                            ^ channels

    print(ashape, bshape)

    for i in range(nsamples):
        a = numpy.loadtxt(directory + "sampled_%05d_spe.dat" % i)
        b = numpy.loadtxt(directory + "sampled_%05d_xrf.dat" % i)
        if a.shape != ashape: raise Exception("Bad dimensions in .../sampled_%05d_spe.dat" % i)
        if b.shape != bshape: raise Exception("Bad dimensions in .../sampled_%05d_xrf.dat" % i)

        targets[i, :] = a[:, 1]
        data[i, :, 0, 0] = b[:, 1]

    # from srxraylib.plot.gol import plot_image
    # plot_image(targets[:,:,0], numpy.arange(nsamples), a[:, 0], aspect='auto', xtitle='nsamples', ytitle='E/eV', title='source spectra')
    # plot_image(data[:,:,0,0], numpy.arange(nsamples), b[:, 0], aspect='auto', xtitle='nsamples', ytitle='E/keV', title='xrf spectra')

    if verbose: print(targets.shape, targets)

    # h5_file = "%s%s_block.h5" % (dir_out, root)
    #
    # f = h5py.File(h5_file, 'r')
    #
    # data = f['allsamples/intensity/stack_data'][:]
    #
    # f.close()

    if verbose: print(data.shape, data)

    size_data = data.shape[0]
    size_target = targets.shape[0]

    if size_data != size_target:
        raise Exception("Data and targets must have the same size.")


    istart_training = 0
    iend_training = int(training_ratio * size_data)
    istart_test = iend_training
    iend_test = size_data

    print(">>>>> iend_training", iend_training)
    if nbin > 0: # normal binning
        return (data[istart_training:iend_training,:,::nbin].copy(), targets[istart_training:iend_training].copy()), \
               (data[istart_test:iend_test,:,::nbin].copy(), targets[istart_test:iend_test].copy())
    else: # take the last part
        return (data[istart_training:iend_training,:,(data.shape[-1]//numpy.abs(nbin)):].copy(), targets[istart_training:iend_training].copy()), \
           (data[istart_test:iend_test,:,(data.shape[-1]//numpy.abs(nbin)):].copy(), targets[istart_test:iend_test].copy())


if __name__ == "__main__":

    do_train = 0

    # model_root = "training_v01"
    # directory = "./data1/"
    # nsamples = 1000

    model_root = "training_v02"
    directory = "./data2/"
    nsamples = 1000


    (training_data, training_target), (test_data, test_target) = load_training_set(nsamples=nsamples,
                                                                                   verbose=1,
                                                                                   nbin=1,
                                                                                   directory=directory) # !!!!!!!!!!!!!! binning  !!!!!!!!!!!

    print("shape of Training: ", training_data.shape, training_target.shape)
    print("shape of Test: ", test_data.shape, test_target.shape)



    # plot_image(training_target[:,:], numpy.arange(training_target.shape[0]), numpy.arange(training_target.shape[1]), aspect='auto', xtitle='nsamples', ytitle='index of E/eV', title='source spectra')
    # plot_image(training_data[:,:,0,0], numpy.arange(training_data.shape[0]), numpy.arange(training_data.shape[1]), aspect='auto', xtitle='nsamples', ytitle='index of E/keV', title='xrf spectra')

    # # data type: images— 4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
    # #            could also be Timeseries data or sequence data— 3D tensors of shape (samples, timesteps, features)
    # #            right now our data is (samples, features (256), timesteps (65))
    # training_data = training_data.reshape((training_data.shape[0], training_data.shape[1], training_data.shape[2], 1))

    #
    min_training_data = training_data.min()
    max_training_data = training_data.max()
    print("Min, Max of Training: ", min_training_data, max_training_data)


    # normalization
    training_data = training_data.astype('float32')
    training_data = (training_data - min_training_data) / (max_training_data - min_training_data)

    test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
    test_data = test_data.astype('float32')
    test_data = (test_data - min_training_data) / (max_training_data - min_training_data)

    #
    #
    #
    # # train_labels = to_categorical(train_labels)
    # # test_labels = to_categorical(test_labels)
    #
    #
    #
    #

    if do_train:
        input_shape = tuple((training_data.shape[1], training_data.shape[2], training_data.shape[3]))
        output_size = training_target.shape[1]
        print("input shape: ", input_shape)
        print("output shape: ", output_size)

        model = get_model(input_shape=input_shape, output_size=output_size)

        # To perform regression toward a vector of continuous values, end your stack of layers
        # with a Dense layer with a number of units equal to the number of values you’re trying
        # to predict (often a single one, such as the price of a house), and no activation. Several
        # losses can be used for regression, most commonly mean_squared_error ( MSE ) and
        # mean_absolute_error ( MAE )

        # to choose thecorrect loss:
        # For instance, you’ll use binary crossentropy for a two-class classification
        # problem, categorical crossentropy for a many-class classification problem, mean-
        # squared error for a regression problem, connectionist temporal classification ( CTC )
        # for a sequence-learning problem, and so on.

        model.compile(
                      # optimizer='rmsprop',
                      optimizer=RMSprop(lr=1e-4),
                      loss='mse',
                      # metrics=['mae'], # mean absolute error
                      # loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      )

        # filename = 'training_v1.csv'
        # import tensorflow as tf
        # history_logger = tf.keras.callbacks.CSVLogger(filename, separator=" ", append=False)


        history = model.fit(training_data, training_target,
                            epochs=1500, batch_size=64, validation_split=0.2,
                            # callbacks=[history_logger],
                            )

        model.save('%s.h5' % model_root)


        history_dict = history.history


        with open("%s.json" % model_root, "w") as outfile:
            json.dump(history_dict, outfile)

    else:
        dir_out = "./"
        model = load_model('%s/%s.h5' % (dir_out, model_root))

        f = open("%s/%s.json" % (dir_out, model_root), "r")
        f_txt = f.read()
        history_dict = json.loads(f_txt)

    print(history_dict.keys())

    import matplotlib.pyplot as plt
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plot(epochs, loss_values,
         epochs, val_loss_values,
         legend=['loss','val_loss'], xtitle='Epochs', ytitle='Loss', show=0)

    # mae_values = history_dict['mae']
    # val_mae_values = history_dict['val_mae']
    # plot(epochs, mae_values,
    #      epochs, val_mae_values,
    #      legend=['mae','val_mae'], xtitle='Epochs', ytitle='mae')

    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plot(epochs, val_acc_values,
         epochs, acc_values,
         legend=['accuracy on validation set','accuracy on training set'],
         color=['g','b'], xtitle='Epochs', ytitle='accuracy')


    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    # plt.plot(epochs, mae_values, 'b', label='mae')
    plt.title('Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    #
    # test evaluation
    #
    test_loss, test_acc = model.evaluate(test_data, test_target)
    #
    print(test_acc)


    #
    # predictions

    predictions = model.predict(test_data)
    print(predictions.shape)

    plot_image(predictions, title="predictions", aspect='auto', show=0)

    plot_image(test_target, title="test target", aspect='auto', show=1)

    for i in range(test_target.shape[0]):
        plot(numpy.arange(predictions.shape[1]), predictions[i],
             numpy.arange(test_target.shape[1]), test_target[i],
             legend=["predicted", "target"], title="index=%d" % i)




