# Whole Image Computation Model Deployment

This is a ros package which can be used to deploy trained models. This package defines ros-service on various
deployment platforms (see details below). Just run the service from this package and you can call it
from your packages. 

The models were trained using : https://github.com/mpkuse/cartwheel_train. You can use the utils in cartwheel_train
to convert the model from keras's .h5 to tensorflow protobuf.

### Folders
- *tx2_py3_tfserver*: A working server with python3 and Tensorflow 1.13+ (1.11 also ok). Can load from .h5 or .pb
- *standalone*: Trying to load protobuf with tensorRT on tx2 (so far not 100% success, some issues)
- *tx2_tensorrt_src*: TensorRT is currently only available on tx2 as C++. This is my attempt to get things to work on tx2 with TensorRT, so far not success.  

I have tried to deploy on
- a) TX2 (with tensorflow + Python3), same code also works on desktop
- b) TensorRT, however it still has issues, see `standalone`
- c) Intel Movidius Neural Compute Stick 2


### Authors
Manohar Kuse <mpkuse@connect.ust.hk>
