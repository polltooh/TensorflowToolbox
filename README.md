# TensorflowToolbox
Some useful functions for using tensorflow

### Install Instruction:

#### Install this Toolbox
Then please make sure the directory is in the python search path.<br>
Replace the </path/to/store> with the dir you want to store
~~~
cd </path/to/store/>
git clone https://github.com/polltooh/TensorflowToolbox.git
export PYTHONPATH=$PYTHONPATH:</path/to/store/>
~~~

Then you should be able to import TensorflowToolbox from anywhere

#### Install tensorflow, current only support version 0.12
~~~
pip install tensorflow-gpu
~~~

#### Install rest of the python package
~~~
sudo pip install -r requirements.txt
~~~


### Please see my other githubs for usage example

video_analysis: https://github.com/polltooh/video_analysis <br>
Fully convolution network for density map estimation or segmentation

IIGAN: https://github.com/polltooh/IIGAN <br>
image to image translation with GAN.

video_lstm: https://github.com/polltooh/video_lstm <br>
Video analysis with lstm network. 

video_dann: https://github.com/polltooh/video_dann <br>
Domain adaptation with adversery training
