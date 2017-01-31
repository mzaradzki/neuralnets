
## Dogs and Cats : Kaggle image recognition


https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition


This folder follows FastAI (Jeremy Howard) guidelines to build a Deep Convolution Network that classifies cats and dogs.

The script uses Keras with Theano backend : see installation steps on main README of the repo.

The script was run on AWS EC2 using P2 instance.

Idea of FastAI MOOC : dont train a full DCN but instead load the configuration of the VGG one, then retrain the last layer.