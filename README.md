# neuralnets

Experimentations with Theano and TensorFlow

These scripts are run on AWS EC2 instance based on AMI (Ireland) :
    cs231n_caffe_torch7_keras_lasagne_v2 (ami-e8a1fe9b)
    
At EC2 configuration time, to setup Jupyter web I follow this tutorial :
    http://efavdb.com/deep-learning-with-jupyter-on-aws/
    
To re-use the same folder across multiple EC2 launches I use AWS EFS :
    ($ sudo apt-get upgrade ?)
    $ sudo apt-get -y install nfs-common
    ($ reboot ?)

The EC2 AMI comes with Theano but TensorFlow needs to be installed :
    ($ sudo easy_install --upgrade pip ?)
    $ pip install tensorflow
    
    