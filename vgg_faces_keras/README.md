

We implement a sequential model in Keras based on the VGG-face model :

http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf

The VGG page has the model weights in .mat format (MatConvNet) so we write a script to transpose them for Keras.

As the VGG Face model works best on image centered on faces we use OpenCV to locate the faces in an image to crop the most relevant section.