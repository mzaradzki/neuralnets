import copy
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Permute, Add, add
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D


def convblock(cdim, nb, bits=3):
	L = []

	for k in range(1, bits + 1):
		convname = 'conv' + str(nb) + '_' + str(k)
		if False:
			# first version I tried
			L.append(ZeroPadding2D((1, 1)))
			L.append(Convolution2D(cdim, kernel_size=(3, 3), activation='relu', name=convname))
		else:
			L.append(Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname))

	L.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	return L


def fcn32_blank(image_size=512):
	withDO = False  # no effect during evaluation but usefull for fine-tuning

	if True:
		mdl = Sequential()

		# First layer is a dummy-permutation = Identity to specify input shape
		mdl.add(Permute((1, 2, 3), input_shape=(image_size, image_size, 3)))  # WARNING : axis 0 is the sample dim

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

		mdl.add(Convolution2D(4096, kernel_size=(7, 7), padding='same', activation='relu', name='fc6'))  # WARNING border
		if withDO:
			mdl.add(Dropout(0.5))
		mdl.add(Convolution2D(4096, kernel_size=(1, 1), padding='same', activation='relu', name='fc7'))  # WARNING border
		if withDO:
			mdl.add(Dropout(0.5))

		# WARNING : model decapitation i.e. remove the classifier step of VGG16 (usually named fc8)

		mdl.add(Convolution2D(21, kernel_size=(1, 1), padding='same', activation='relu', name='score_fr'))

		convsize = mdl.layers[-1].output_shape[2]
		deconv_output_size = (convsize - 1) * 2 + 4  # INFO: =34 when images are 512x512
		# WARNING : valid, same or full ?
		mdl.add(Deconvolution2D(21, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation=None, name='score2'))

		extra_margin = deconv_output_size - convsize * 2  # INFO: =2 when images are 512x512
		assert (extra_margin > 0)
		assert (extra_margin % 2 == 0)
		# INFO : cropping as deconv gained pixels
		# print(extra_margin)
		c = ((0, extra_margin), (0, extra_margin))
		# print(c)
		mdl.add(Cropping2D(cropping=c))
		# print(mdl.summary())

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
	if fcn32model is None:
		fcn32model = fcn32_blank()

	fcn32shape = fcn32model.layers[-1].output_shape
	assert (len(fcn32shape) == 4)
	assert (fcn32shape[0] is None)  # batch axis
	assert (fcn32shape[3] == 21)  # number of filters
	assert (fcn32shape[1] == fcn32shape[2])  # must be square

	fcn32size = fcn32shape[1]  # INFO: =32 when images are 512x512

	if fcn32size != 32:
		print('WARNING : handling of image size different from 512x512 has not been tested')

	sp4 = Convolution2D(21, kernel_size=(1, 1), padding='same', activation=None, name='score_pool4')

	# INFO : to replicate MatConvNet.DAGN.Sum layer see documentation at :
	# https://keras.io/getting-started/sequential-model-guide/
	summed = add(inputs=[sp4(fcn32model.layers[14].output), fcn32model.layers[-1].output])

	# INFO :
	# final 16x16 upsampling of "summed" using deconv layer upsample_new (32, 32, 21, 21)
	# deconv setting is valid if (528-32)/16 + 1 = deconv_input_dim (= fcn32size)
	deconv_output_size = (fcn32size - 1) * 16 + 32  # INFO: =528 when images are 512x512
	upnew = Deconvolution2D(21, kernel_size=(32, 32),
							padding='valid',  # WARNING : valid, same or full ?
							strides=(16, 16),
							activation=None,
							name='upsample_new')

	extra_margin = deconv_output_size - fcn32size * 16  # INFO: =16 when images are 512x512
	assert (extra_margin > 0)
	assert (extra_margin % 2 == 0)
	# print(extra_margin)
	# INFO : cropping as deconv gained pixels
	crop_margin = Cropping2D(cropping=((0, extra_margin), (0, extra_margin)))

	return Model(fcn32model.input, crop_margin(upnew(summed)))


def prediction(kmodel, crpimg, transform=False):
	# INFO : crpimg should be a cropped image of the right dimension

	# transform=True seems more robust but I think the RGB channels are not in right order

	imarr = np.array(crpimg).astype(np.float32)

	if transform:
		imarr[:, :, 0] -= 129.1863
		imarr[:, :, 1] -= 104.7624
		imarr[:, :, 2] -= 93.5940
		#
		# WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
		aux = copy.copy(imarr)
		imarr[:, :, 0] = aux[:, :, 2]
		imarr[:, :, 2] = aux[:, :, 0]

	# imarr[:,:,0] -= 129.1863
	# imarr[:,:,1] -= 104.7624
	# imarr[:,:,2] -= 93.5940

	# imarr = imarr.transpose((2, 0, 1))
	imarr = np.expand_dims(imarr, axis=0)

	return kmodel.predict(imarr)


if __name__ == "__main__":
	md = fcn32_blank()
	md = fcn_32s_to_16s(md)
	print(md.summary())
