import copy
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Permute
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import merge


def convblock(cdim, nb, bits=3):
    L = []
    
    for k in range(1,bits+1):
        convname = 'conv'+str(nb)+'_'+str(k)
        if False:
            # first version I tried
            L.append( ZeroPadding2D((1, 1)) )
            L.append( Convolution2D(cdim, 3, 3, activation='relu', name=convname) )
        else:
            L.append( Convolution2D(cdim, 3, 3, border_mode='same', activation='relu', name=convname) )
    
    L.append( MaxPooling2D((2, 2), strides=(2, 2)) )
    
    return L


def fcn32_blank():
    
    withDO = False # no effect during evaluation but usefull for fine-tuning
    
    if True:
        mdl = Sequential()
        
        # First layer is a dummy-permutation = Identity to specify input shape
        #mdl.add( Permute((1,2,3), input_shape=(3,224,224)) ) # WARNING : 0 is the sample dim
        mdl.add( Permute((1,2,3), input_shape=(3,512,512)) ) # WARNING : 0 is the sample dim

        for l in convblock(64, 1, bits=2):
            mdl.add(l)

        for l in convblock(128, 2, bits=2):
            mdl.add(l)
        
        for l in convblock(256, 3, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 4, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 5, bits=3):
            mdl.add(l)
        
        mdl.add( Convolution2D(4096, 7, 7, border_mode='same', activation='relu', name='fc6') ) # WARNING border
        if withDO:
            mdl.add( Dropout(0.5) )
        mdl.add( Convolution2D(4096, 1, 1, border_mode='same', activation='relu', name='fc7') ) # WARNING border
        if withDO:
            mdl.add( Dropout(0.5) )
        
        # WARNING : model decapitation i.e. remove the classifier step of VGG16 (usually named fc8)
        
        mdl.add( Convolution2D(21, 1, 1,
                               border_mode='same', # WARNING : zero or same ? does not matter for 1x1
                               activation='relu', name='score_fr') )
        
        mdl.add( Deconvolution2D(21, 4, 4,
                                 output_shape=(None, 21, 34, 34),
                                 subsample=(2, 2),
                                 border_mode='valid', # WARNING : valid, same or full ?
                                 activation=None,
                                 name = 'score2') )
        
        mdl.add( Cropping2D(cropping=((1, 1), (1, 1))) ) # WARNING : cropping as deconv gained pixels
        
        return mdl
    
    else:
        # See following link for a version based on Keras functional API :
        # gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
        raise ValueError('not implemented')



# WARNING : explanation about Deconvolution2D layer
# http://stackoverflow.com/questions/39018767/deconvolution2d-layer-in-keras
# the code example in the help (??Deconvolution2D) is very usefull too
# ?? Deconvolution2D

def fcn_32s_to_16s(fcn32model=None):
    
    if (fcn32model is None):
        fcn32model = fcn32_blank()
    
    sp4 = Convolution2D(21, 1, 1,
                    border_mode='same', # WARNING : zero or same ? does not matter for 1x1
                    activation=None, # WARNING : to check
                    name='score_pool4')

    # INFO : to replicate MatConvNet.DAGN.Sum layer see documentation at :
    # https://keras.io/getting-started/sequential-model-guide/
    summed = merge([sp4(fcn32model.layers[14].output), fcn32model.layers[-1].output], mode='sum')

    # INFO : final 16x16 upsampling of "summed" using deconv layer upsample_new (32, 32, 21, 21)
    upnew = Deconvolution2D(21, 32, 32,
                            output_shape=(None, 21, 528, 528),
                            border_mode='valid', # WARNING : valid, same or full ?
                            subsample=(16, 16),
                            activation=None,
                            name = 'upsample_new')

    crop8 = Cropping2D(cropping=((8, 8), (8, 8))) # WARNING : cropping as deconv gained pixels

    return Model(fcn32model.input, crop8(upnew(summed))) # fcn16model


def fcn_32s_to_8s(fcn32model=None):
    
    if (fcn32model is None):
        fcn32model = fcn32_blank()
    
    sp4 = Convolution2D(21, 1, 1,
                    border_mode='same', # WARNING : zero or same ? does not matter for 1x1
                    activation=None, # WARNING : to check
                    name='score_pool4')

    # INFO : to replicate MatConvNet.DAGN.Sum layer see documentation at :
    # https://keras.io/getting-started/sequential-model-guide/
    score_fused = merge([sp4(fcn32model.layers[14].output), fcn32model.layers[-1].output], mode='sum')

    s4deconv = Deconvolution2D(21, 4, 4,
                            output_shape=(None, 21, 66, 66),
                            border_mode='valid', # WARNING : valid, same or full ?
                            subsample=(2, 2),
                            activation=None,
                            name = 'score4')

    crop1111 = Cropping2D(cropping=((1, 1), (1, 1))) # WARNING : cropping as deconv gained pixels

    score4 = crop1111(s4deconv(score_fused))

    # WARNING : check dimensions
    sp3 = Convolution2D(21, 1, 1,
                        border_mode='same', # WARNING : zero or same ? does not matter for 1x1
                        activation=None, # WARNING : to check
                        name='score_pool3')

    score_final = merge([sp3(fcn32model.layers[10].output), score4], mode='sum') # WARNING : is that correct ?

    upsample = Deconvolution2D(21, 16, 16,
                            output_shape=(None, 21, 520, 520),
                            border_mode='valid', # WARNING : valid, same or full ?
                            subsample=(8, 8),
                            activation=None,
                            name = 'upsample')

    bigscore = upsample(score_final)

    crop4444 = Cropping2D(cropping=((4, 4), (4, 4))) # WARNING : cropping as deconv gained pixels

    coarse = crop4444(bigscore)

    return Model(fcn32model.input, coarse) # fcn8model


def prediction(kmodel, crpimg, transform=False):
    
    # INFO : crpimg should be a cropped image of the right dimension
    
    # transform=True seems more robust but I think the RGB channels are not in right order
    
    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:,:,0] -= 129.1863
        imarr[:,:,1] -= 104.7624
        imarr[:,:,2] -= 93.5940
        #
        # WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
        aux = copy.copy(imarr)
        imarr[:, :, 0] = aux[:, :, 2]
        imarr[:, :, 2] = aux[:, :, 0]

        #imarr[:,:,0] -= 129.1863
        #imarr[:,:,1] -= 104.7624
        #imarr[:,:,2] -= 93.5940

    imarr = imarr.transpose((2,0,1))
    imarr = np.expand_dims(imarr, axis=0)

    return kmodel.predict(imarr)