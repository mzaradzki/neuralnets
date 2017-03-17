

Implement the **VGG-Face** model using **Keras** a sequential model.

My post on Medium explaining this :
* https://medium.com/@m.zaradzki/face-recognition-with-keras-and-opencv-2baf2a83b799#

References from the authors of the model:
* http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
* http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf

The VGG page has the model weights in .mat format (MatConvNet) so we write a script to transpose them for Keras.

As the VGG Face model works best on image centered on faces we use **OpenCV** to locate the faces in an image to crop the most relevant section.

We use Keras Functional API to derive a face-feature-vector model based on the logit layer (second-last). This model can be used to check if two images are represent the same faces even if the faces are not part of the training sample. This is done using hte cosine similarity of the face feature vectors.