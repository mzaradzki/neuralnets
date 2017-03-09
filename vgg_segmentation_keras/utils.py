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


def fcn32_blank(image_size=512):
    
    withDO = False # no effect during evaluation but usefull for fine-tuning
    
    if True:
        mdl = Sequential()
        
        # First layer is a dummy-permutation = Identity to specify input shape
        mdl.add( Permute((1,2,3), input_shape=(3,image_size,image_size)) ) # WARNING : axis 0 is the sample dim

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
        
        convsize = mdl.layers[-1].output_shape[2]
        deconv_output_size = (convsize-1)*2+4 # INFO: =34 when images are 512x512
        mdl.add( Deconvolution2D(21, 4, 4,
                                 output_shape=(None, 21, deconv_output_size, deconv_output_size),
                                 subsample=(2, 2),
                                 border_mode='valid', # WARNING : valid, same or full ?
                                 activation=None,
                                 name = 'score2') )
        
        extra_margin = deconv_output_size - convsize*2 # INFO: =2 when images are 512x512
        assert (extra_margin > 0)
        assert (extra_margin % 2 == 0)
        mdl.add( Cropping2D(cropping=((extra_margin/2, extra_margin/2),
                                      (extra_margin/2, extra_margin/2))) ) # INFO : cropping as deconv gained pixels
        
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
        
    fcn32shape = fcn32model.layers[-1].output_shape
    assert (len(fcn32shape) == 4)
    assert (fcn32shape[0] is None) # batch axis
    assert (fcn32shape[1] == 21) # number of filters
    assert (fcn32shape[2] == fcn32shape[3]) # must be square
    
    fcn32size = fcn32shape[2] # INFO: =32 when images are 512x512
    
    if (fcn32size != 32):
        print('WARNING : handling of image size different from 512x512 has not been tested')
    
    sp4 = Convolution2D(21, 1, 1,
                    border_mode='same', # WARNING : zero or same ? does not matter for 1x1
                    activation=None, # WARNING : to check
                    name='score_pool4')

    # INFO : to replicate MatConvNet.DAGN.Sum layer see documentation at :
    # https://keras.io/getting-started/sequential-model-guide/
    summed = merge([sp4(fcn32model.layers[14].output), fcn32model.layers[-1].output], mode='sum')

    # INFO :
    # final 16x16 upsampling of "summed" using deconv layer upsample_new (32, 32, 21, 21)
    # deconv setting is valid if (528-32)/16 + 1 = deconv_input_dim (= fcn32size)
    deconv_output_size = (fcn32size-1)*16+32 # INFO: =528 when images are 512x512
    upnew = Deconvolution2D(21, 32, 32,
                            output_shape=(None, 21, deconv_output_size, deconv_output_size),
                            border_mode='valid', # WARNING : valid, same or full ?
                            subsample=(16, 16),
                            activation=None,
                            name = 'upsample_new')

    extra_margin = deconv_output_size - fcn32size*16 # INFO: =16 when images are 512x512
    assert (extra_margin > 0)
    assert (extra_margin % 2 == 0)
    crop_margin = Cropping2D(cropping=((extra_margin/2, extra_margin/2),
                                       (extra_margin/2, extra_margin/2))) # INFO : cropping as deconv gained pixels

    return Model(fcn32model.input, crop_margin(upnew(summed)))


def fcn_32s_to_8s(fcn32model=None):
    
    if (fcn32model is None):
        fcn32model = fcn32_blank()
        
    fcn32shape = fcn32model.layers[-1].output_shape
    assert (len(fcn32shape) == 4)
    assert (fcn32shape[0] is None) # batch axis
    assert (fcn32shape[1] == 21) # number of filters
    assert (fcn32shape[2] == fcn32shape[3]) # must be square
    
    fcn32size = fcn32shape[2] # INFO: =32 when images are 512x512
    
    if (fcn32size != 32):
        print('WARNING : handling of image size different from 512x512 has not been tested')
    
    sp4 = Convolution2D(21, 1, 1,
                    border_mode='same', # WARNING : zero or same ? does not matter for 1x1
                    activation=None, # WARNING : to check
                    name='score_pool4')

    # INFO : to replicate MatConvNet.DAGN.Sum layer see documentation at :
    # https://keras.io/getting-started/sequential-model-guide/
    score_fused = merge([sp4(fcn32model.layers[14].output), fcn32model.layers[-1].output], mode='sum')

    deconv4_output_size = (fcn32size-1)*2+4 # INFO: =66 when images are 512x512
    s4deconv = Deconvolution2D(21, 4, 4,
                            output_shape=(None, 21, deconv4_output_size, deconv4_output_size),
                            border_mode='valid', # WARNING : valid, same or full ?
                            subsample=(2, 2),
                            activation=None,
                            name = 'score4')

    extra_margin4 = deconv4_output_size - fcn32size*2 # INFO: =2 when images are 512x512
    assert (extra_margin4 > 0)
    assert (extra_margin4 % 2 == 0)
    crop_margin4 = Cropping2D(cropping=((extra_margin4/2, extra_margin4/2),
                                        (extra_margin4/2, extra_margin4/2))) # INFO : cropping as deconv gained pixels

    score4 = crop_margin4(s4deconv(score_fused))

    # WARNING : check dimensions
    sp3 = Convolution2D(21, 1, 1,
                        border_mode='same', # WARNING : zero or same ? does not matter for 1x1
                        activation=None, # WARNING : to check
                        name='score_pool3')

    score_final = merge([sp3(fcn32model.layers[10].output), score4], mode='sum') # WARNING : is that correct ?

    assert (fcn32size*2 == fcn32model.layers[10].output_shape[2])
    deconvUP_output_size = (fcn32size*2-1)*8+16 # INFO: =520 when images are 512x512
    upsample = Deconvolution2D(21, 16, 16,
                            output_shape=(None, 21, deconvUP_output_size, deconvUP_output_size),
                            border_mode='valid', # WARNING : valid, same or full ?
                            subsample=(8, 8),
                            activation=None,
                            name = 'upsample')

    bigscore = upsample(score_final)

    extra_marginUP = deconvUP_output_size - (fcn32size*2)*8 # INFO: =8 when images are 512x512
    assert (extra_marginUP > 0)
    assert (extra_marginUP % 2 == 0)
    crop_marginUP = Cropping2D(cropping=((extra_marginUP/2, extra_marginUP/2),
                                         (extra_marginUP/2, extra_marginUP/2))) # INFO : cropping as deconv gained pixels

    coarse = crop_marginUP(bigscore)

    return Model(fcn32model.input, coarse)


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