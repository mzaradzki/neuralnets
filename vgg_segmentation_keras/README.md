
Implements and tests the **FCN-16s** and **FCN-8s** models for image segmentation using **Keras** deep-learning library.

My post on Medium explaining the model architecture and its Keras implementation :
* https://medium.com/@m.zaradzki/image-segmentation-with-neural-net-d5094d571b1e#


References from the authors of the model:
* title: **Fully Convolutional Networks for Semantic Segmentation**
* authors: **Jonathan Long, Evan Shelhamer, Trevor Darrell**
* link: http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf


### Extracts from the article relating to the model architecture

**remark**: The model is derived from VGG16.

**remark** : Deconvolution and conv-transpose are synonyms, they perform up-sampling.

#### 4.1. From classifier to dense FCN

We decapitate each net by discarding the final classifier layer [**code comment** : *this is why fc8 is not included*], and convert all fully connected layers to convolutions.

We append a 1x1 convolution with channel dimension 21 [**code comment** : *layer named score_fr*] to predict scores for each of the PASCAL classes (including background) at each of the coarse output locations, followed by a deconvolution layer to bilinearly upsample the coarse outputs to pixel-dense outputs as described in Section 3.3.


#### 4.2. Combining what and where
We define a new fully convolutional net (FCN) for segmentation that combines layers of the feature hierarchy and
refines the spatial precision of the output.
While fully convolutionalized classifiers can be fine-tuned to segmentation as shown in 4.1, and even score highly on the standard metric, their output is dissatisfyingly coarse.
The 32 pixel stride at the final prediction layer limits the scale of detail in the upsampled output.

We address this by adding skips that combine the final prediction layer with lower layers with finer strides.
This turns a line topology into a DAG [**code comment** : *this is why some latter stage layers have 2 inputs*], with edges that skip ahead from lower layers to higher ones.
As they see fewer pixels, the finer scale predictions should need fewer layers, so it makes sense to make them from shallower net outputs.
Combining fine layers and coarse layers lets the model make local predictions that respect global structure.

We first divide the output stride in half by predicting from a 16 pixel stride layer.
We add a 1x1 convolution layer on top of pool4 [**code comment** : *the score_pool4_filter layer*] to produce additional class predictions.
We fuse this output with the predictions computed on top of conv7 (convolutionalized fc7) at stride 32 by adding a 2x upsampling layer and summing [**code comment** : *layer named sum*] both predictions [**code warning** : *requires first layer crop to insure the same size*].

Finally, the stride 16 predictions are upsampled back to the image [**code comment** : *layer named upsample_new*].

We call this net FCN-16s.