#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os

############# setting seed ################
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import random
random.seed(1)
os.environ['PYTHONHASHSEED'] = '0'
###########################################

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import keras
import argparse
#os.environ['LD_LIBRARY_PATH'] = os.getcwd()
from six.moves import range
import sys
import glob
import h5py
import numpy as np
import time
import math
import argparse
import random
import psutil
import socket
import time
from tensorflow.python.client import timeline
from keras.callbacks import CallbackList
import analysis.utils.GANutils as gan # some common functions for gan


################################################
path = 'fifopipe'
def attachFMA_MP_FP32_WU_BN():
    fifo = open(path, 'w')
    fifo.write('B')
    fifo.close()
    time.sleep(50)

def detach():
    fifo = open(path, 'w')
    fifo.write('G')
    fifo.close()
    time.sleep(20)

def attachFMA_BF16():
    fifo = open(path, 'w')
    fifo.write('A')
    fifo.close()
    time.sleep(20)
################################################


#import setGPU #if Caltech
def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception


def BitFlip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

def DivideFiles(FileSearch="/data/LCD/*/*.h5", nEvents=200000, EventsperFile = 10000, Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):

    Files =sorted( glob.glob(FileSearch))
    Filesused = int(math.ceil(nEvents/EventsperFile))
    FileCount=0

    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")

        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]

        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    out=[]
    for j in range(len(Fractions)):
        out.append([])

    SampleI=len(Samples.keys())*[int(0)]

    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName][:Filesused]
        NFiles=len(Sample)

        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI
    return out

# This functions loads data from a file and also does any pre processing
def GetData(datafile, xscale =1, yscale = 100, dimensions = 3, keras_dformat="channels_last"):
    #get data for training
    #if hvd.rank()==0:
    #    print('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')

    X=np.array(f.get('ECAL'))

    Y=f.get('target')
    Y=(np.array(Y[:,1]))

    X[X < 1e-6] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    if dimensions == 2:
        X = np.sum(X, axis=(1))
    X = xscale * X

    Y = Y.astype(np.float32)
    Y = Y/yscale
    if keras_dformat !='channels_last':
       X =np.moveaxis(X, -1, 1)
       ecal = np.sum(X, axis=(2, 3, 4))
    else:
       ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, ecal


def GetEcalFit(sampled_energies, mod=1, xscale=1):
    if mod==0:
      return np.multiply(2, sampled_energies)
    elif mod==1:
      root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
      ratio = np.polyval(root_fit, sampled_energies)
      return np.multiply(ratio, sampled_energies) * xscale


def genbatches(a,n):
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n]


def randomize(a, b, c):
    assert a.shape[0] == b.shape[0]
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    shuffled_c = c[permutation]
    return shuffled_a, shuffled_b, shuffled_c

def GanTrain(discriminator, generator, opt,run_options, run_metadata, global_batch_size,
             warmup_epochs, datapath, EventsperFile, nEvents, WeightsDir, resultfile,
             energies,mod=0, nb_epochs=30, batch_size=128, latent_size=128, gen_weight=6,
             aux_weight=0.2, ecal_weight=0.1, lr=0.001, rho=0.9, decay=0.0,
             g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_',
             xscale=1, verbose=True, keras_dformat='channels_last', analysis=True,
             load_previous_model=False, generator_model=None, discriminator_model=None, last_epoch=0,
             generator_name=None, discriminator_name=None, mp_use=False, train_history_file='default.pkl'):
    start_init = time.time()
    verbose = False

    if (load_previous_model):
        print('[INFO] Loading complete generator model')
        generator = load_model(generator_model)
        print('[INFO] Loading complete discriminator model')
        discriminator = load_model(discriminator_model)
    else:
        print('[INFO] Building discriminator')
        discriminator.compile(
            optimizer=opt,
            loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
            loss_weights=[gen_weight, aux_weight, ecal_weight],options=run_options,run_metadata=run_metadata
        )

        #build the generator
        print('[INFO] Building generator')

        generator.compile(
            optimizer=opt,
            loss='binary_crossentropy',options=run_options,run_metadata=run_metadata
        )

    # build combined Model
    latent = Input(shape=(latent_size, ), name='combined_z')
    fake_image = generator(latent)
    discriminator.trainable = False
    fake, aux, ecal = discriminator(fake_image)
    combined = Model(input=[latent], output=[fake, aux, ecal], name='combined_model')

    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, ecal_weight],options=run_options,run_metadata=run_metadata
    )
    discriminator.trainable = True # workaround for a k2 bug

#    ############# In case we need to relaunch a training process ##############
#    prev_gweights = '/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_no_horovod_pin_mp/generator_params_generator_epoch_013.hdf5'
#    prev_dweights = '/gpfs/scratch/bsc28/bsc28459/3dgan_data/weights_no_horovod_pin_mp/discriminator_params_generator_epoch_013.hdf5'
#    generator.load_weights(prev_gweights)
#    print('Generator initialized from {}'.format(prev_gweights))
#    discriminator.load_weights(prev_dweights)
#    print('Discriminator initialized from {}'.format(prev_dweights))
#    ###########################################################################

    # Getting Data
    Trainfiles, Testfiles = DivideFiles(datapath, nEvents=nEvents, EventsperFile = EventsperFile, datasetnames=["ECAL"], Particles =["Ele"])
    print(Trainfiles)
    print(Testfiles)
    print("Train files: {0} \nTest files: {1}".format(Trainfiles, Testfiles))

    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ecal_test = GetData(dtest, keras_dformat=keras_dformat, xscale=xscale)
       else:
           X_temp, Y_temp, ecal_temp = GetData(dtest, keras_dformat=keras_dformat, xscale=xscale)
           X_test = np.concatenate((X_test, X_temp))
           Y_test = np.concatenate((Y_test, Y_temp))
           ecal_test = np.concatenate((ecal_test, ecal_temp))

    for index, dtrain in enumerate(Trainfiles):
        if index == 0:
            X_train, Y_train, ecal_train = GetData(dtrain, keras_dformat=keras_dformat, xscale=xscale)
        else:
            X_temp, Y_temp, ecal_temp = GetData(dtrain, keras_dformat=keras_dformat, xscale=xscale)
            X_train = np.concatenate((X_train, X_temp))
            Y_train = np.concatenate((Y_train, Y_temp))
            ecal_train = np.concatenate((ecal_train, ecal_temp))

    print("On hostname {0} - After init using {1} memory".format(socket.gethostname(), psutil.Process(os.getpid()).memory_info()[0]))

    nb_test = X_test.shape[0]
    assert X_train.shape[0] == EventsperFile * len(Trainfiles), "# Total events in training files"
    nb_train = X_train.shape[0]# Total events in training files
    total_batches = int(nb_train / global_batch_size)
    print('Total Training batches = {} with {} events'.format(total_batches, nb_train))

    train_history = defaultdict(list)

    print('Initialization time was {} seconds'.format(time.time() - start_init))
##################################################################
    if (mp_use):
        attachFMA_MP_FP32_WU_BN()
##################################################################
    for epoch in range(last_epoch,nb_epochs):
        epoch_start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        sys.stdout.flush()

        randomize(X_train, Y_train, ecal_train)

        epoch_gen_loss = []
        epoch_disc_loss = []

        image_batches = genbatches(X_train, batch_size)
        energy_batches = genbatches(Y_train, batch_size)
        ecal_batches = genbatches(ecal_train, batch_size)

        for index in range(total_batches):
            start = time.time()
            image_batch = next(image_batches)
            energy_batch = next(energy_batches)
            ecal_batch = next(ecal_batches)

            noise = np.random.normal(0, 1, (batch_size, latent_size))
            sampled_energies = np.random.uniform(0.1, 5,( batch_size, 1))
            generator_ip = np.multiply(sampled_energies, noise)
            # ecal sum from fit
            ecal_ip = GetEcalFit(sampled_energies, mod, xscale)

            generated_images = generator.predict(generator_ip, verbose=0)
            real_batch_loss = discriminator.train_on_batch(image_batch, [BitFlip(np.ones(batch_size)), energy_batch, ecal_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [BitFlip(np.zeros(batch_size)), sampled_energies, ecal_ip])
            #print('real batch loss ={}'.format(real_batch_loss))
            #print('fake batch loss ={}'.format(fake_batch_loss))
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            trick = np.ones(batch_size)
            gen_losses = []
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_energies = np.random.uniform(0.1, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = GetEcalFit(sampled_energies, mod, xscale)
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))
            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])

            if (index % 100)==0: # and hvd.rank()==0:
                # progress_bar.update(index)
                print('processed {}/{} batches in {}'.format(index + 1, total_batches, time.time() - start))
                sys.stdout.flush()

        # save weights every epoch
        safe_mkdir(WeightsDir)

        print ("saving weights of gen")
        generator.save_weights(WeightsDir + '/generator_{0}{1:03d}.hdf5'.format(g_weights, epoch), overwrite=True)

        print ("saving weights of disc")
        discriminator.save_weights(WeightsDir + '/discriminator_{0}{1:03d}.hdf5'.format(d_weights, epoch), overwrite=True)

        epoch_time = time.time()-epoch_start
        print("The {} epoch took {} seconds".format(epoch, epoch_time))

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        pickle.dump({'train': train_history}, open(train_history_file, 'wb'))

        print('{0:<22s} | {1:4s} | {2:15s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))

    print ("Saving complete generator model")
    generator.save(WeightsDir + '/' + generator_name + '.h5')

    print ("Saving complete discriminator model")
    discriminator.save(WeightsDir + '/' + discriminator_name + '.h5')

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--model', '-m', action='store', type=str, default='EcalEnergyGan', help='Model architecture to use.')
    parser.add_argument('--nbepochs', action='store', type=int, default=25, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=200, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/eos/project/d/dshep/LCD/V1/*scan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Number of Data points to use')
    parser.add_argument('--nbperfile', action='store', type=int, default=10000, help='Number of events in a file.')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--weightsdir', action='store', type=str, default='/gkhattak/hvdweights/', help='Directory to store weights.')
    parser.add_argument('--mod', action='store', type=int, default=1, help='How to calculate Ecal sum corressponding to\
                        energy.\n [0].. factor 50 \n[1].. Fit from Root')
    parser.add_argument('--xscale', action='store', type=int, default=100, help='Multiplication factor for ecal deposition')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--learningRate', '-lr', action='store', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--optimizer', action='store', type=str, default='RMSprop', help='Keras Optimizer to use.')
    parser.add_argument('--intraop', action='store', type=int, default=9, help='Sets config.intra_op_parallelism_threads and OMP_NUM_THREADS')
    parser.add_argument('--interop', action='store', type=int, default=1, help='Sets config.inter_op_parallelism_threads')
    parser.add_argument('--warmupepochs', action='store', type=int, default=5, help='No warmup epochs')
    parser.add_argument('--channel_format', action='store', type=str, default='channels_last', help='NCHW vs NHWC')
    parser.add_argument('--analysis', action='store', type=bool, default=False, help='Calculate optimisation function')
    parser.add_argument('--load_previous_model', action='store_true', help='To restart a previous training process')
    parser.add_argument('--generator_model', action='store', type=str, default=None, help='Path to the generator model to load, works just if \
                        --load_previous_model is set to True')
    parser.add_argument('--discriminator_model', action='store', type=str, default=None, help='Path to the discriminator model to load, works \
                        just if --load_previous_model is set to True')
    parser.add_argument('--last_epoch', action='store', type=int, default=0, help='The value of the last processed epoch to continue the training \
                        process')
    parser.add_argument('--generator_name', action='store', type=str, default='./generator_model.h5', help='PATH to store the complete generator \
                        model')
    parser.add_argument('--discriminator_name', action='store', type=str, default='./discriminator_model.h5', help='PATH to store the complete \
                        discriminator model')
    parser.add_argument('--mp_use', action='store_true', help='Use this flag to enable Mixed Precision use')
    parser.add_argument('--train_history_file', action='store', type=str, default='./train_history_file.pkl', help='PATH to store the training history')
    return parser

if __name__ == '__main__':

    import keras.backend as K

    from keras.layers import Input
    from keras.models import Model
    from keras.optimizers import Adadelta, Adam, RMSprop
    from keras.utils.generic_utils import Progbar
    from sklearn.model_selection import train_test_split
    from keras.models import load_model

    import tensorflow as tf

    #Values to be set by user
    parser = get_parser()
    params = parser.parse_args()
    print(params)

    d_format = params.channel_format

    if d_format == 'channels_first':
        print('Setting th channel ordering (NCHW)')
        K.set_image_data_format('channels_first')
#        K.set_image_dim_ordering('th')
    else:
        print('Setting tf channel ordering (NHWC)')
        K.set_image_data_format('channels_last')
#        K.set_image_dim_ordering('tf')

    #config = tf.compat.v1.ConfigProto()#(log_device_placement=True)
    config = tf.ConfigProto()#(log_device_placement=True)
    config.intra_op_parallelism_threads = params.intraop
    config.inter_op_parallelism_threads = params.interop
    os.environ['KMP_BLOCKTIME'] = str(1)
    os.environ['KMP_SETTINGS'] = str(1)
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact'
    # os.environ['KMP_AFFINITY'] = 'balanced'
    os.environ['OMP_NUM_THREADS'] = str(params.intraop)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)
    #tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    K.set_session(tf.Session(config=config))
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    #Architectures to import
    from EcalEnergyGan import generator, discriminator

    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
        # Analysis
    analysis = params.analysis # if analysing
    energies = [100, 200, 300, 400] # Bins
    resultfile = '3dgan_analysis_mp_no_horovod.pkl' # analysis result

    global_batch_size = batch_size #* hvd.size()
    print("Global batch size is: {0} / batch size is: {1}".format(global_batch_size, batch_size))
    sys.stdout.flush()
    latent_size = params.latentsize #latent vector size
    verbose = params.verbose
    datapath = params.datapath#Data path on EOS CERN
    EventsperFile = params.nbperfile#Events in a file
    nEvents = params.nbEvents#Total events for training
    fitmod = params.mod
    weightdir = params.weightsdir
    xscale = params.xscale
    warmup_epochs = params.warmupepochs
    
    opt = getattr(keras.optimizers, params.optimizer)
    #opt = RMSprop()
    opt = opt(params.learningRate)# * hvd.size())
    # Building discriminator and generator
    d = discriminator(keras_dformat=d_format)
    g = generator(latent_size, keras_dformat=d_format)
    ######### Start the training process ###########
    GanTrain(d, g, opt, run_options, run_metadata, global_batch_size, warmup_epochs,
             datapath, EventsperFile, nEvents, weightdir, resultfile, energies,
             mod=fitmod, nb_epochs=nb_epochs, batch_size=batch_size,
             latent_size=latent_size, gen_weight=8, aux_weight=0.2, ecal_weight=0.1,
             xscale = xscale, verbose=verbose, keras_dformat=d_format, analysis=analysis,
             load_previous_model=params.load_previous_model,
             generator_model=params.generator_model,
             discriminator_model=params.discriminator_model,
             last_epoch=params.last_epoch, generator_name=params.generator_name,
             discriminator_name=params.discriminator_name, mp_use=params.mp_use,
             train_history_file=params.train_history_file)
#    #################################################
#    to = timeline.Timeline(run_metadata.step_stats)
#    trace = to.generate_chrome_trace_format()
#    with open('full_train_trace_mp.json', 'w') as out:
#          out.write(trace)

