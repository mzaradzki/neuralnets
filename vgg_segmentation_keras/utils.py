
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Permute
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D


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