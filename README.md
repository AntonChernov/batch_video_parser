## Yolo4 Installation
Full instruction here:

https://robocademy.com/2020/05/01/a-gentle-introduction-to-yolo-v4-for-object-detection-in-ubuntu-20-04/

Quick instruction:


1 Install cmake if not installed: 
```sudo apt install cmake``` 

CUDA 10.0 (For GPU)
You can install CUDA if you have a GPU from NVIDIA, adding GPU for YOLO will speed up the object detection process.

CUDA is a parallel computing platform and application programming interface model created by Nvidia. It allows software developers and software engineers to use a CUDA-enabled graphics processing unit for general purpose processing.

Here are the tutorials to install CUDA 10 on Ubuntu.

Download and install CUDA 10 Toolkit (https://developer.nvidia.com/cuda-toolkit-archive)

How to install CUDA on Ubuntu 20.04 Focal Fossa Linux(https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux)

2 Install OpenCV - ```sudo apt install libopencv-dev python3-opencv``` or install all lib from requirements.txt

3 cuDNN >= 7.0 for CUDA 10.0 (for GPU)
The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.

Here is the installation procedure for cuDNN.

Installing cuDNN On Linux(https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_741/cudnn-install/index.html)

4 OpenMP (for CPU)
OpenMP uses a portable, scalable model that gives programmers a simple and flexible interface for developing parallel applications for platforms ranging from the standard desktop computer to the supercomputer.

Here is the command to install OpenMP in Ubuntu 20.04:

```sudo apt install libomp-dev```    

5 Clone next repo: https://github.com/pjreddie/darknet


## Build 

Building YOLO v4 using Make

Switch to the darknet folder after download. 
Open the Makefile in the darknet folder. 
You can see some of the variables at the beginning of the Makefile. 
If you want to compile darknet for CPU, you can use the following flags.


For CPU build

Set AVX=1 and OPENMP=1 to speedup on CPU (if an error occurs then set AVX=0),
Set LIBSO=1 will create the shared library of the darknet, ‘libdarknet.so‘, which is used to interface darknet and Python.
Set ZED_CAMERA=1 if you are working with ZED camera and its SDK

```
GPU=0

CUDNN=0

CUDNN_HALF=0

OPENCV=1

AVX=1

OPENMP=1

LIBSO=1  

ZED_CAMERA=0

ZED_CAMERA_v2_8=0
```

For GPU build

set GPU=1 and CUDNN=1 to speedup on GPU

set CUDNN_HALF=1 to further speedup 3 x times (Mixed-precision on Tensor Cores) GPU: Volta, Xavier, Turing and higher

```
GPU=1

CUDNN=1

CUDNN_HALF=1

OPENCV=1

AVX=0

OPENMP=0

LIBSO=1  

ZED_CAMERA=0 

ZED_CAMERA_v2_8=0
```

After doing these changes, just execute the following command from the darknet folder.

`make`

After build, you can able find _**darknet**_ and _**libdarknet.so**_ in the build path.

## Testing YOLO v4
After building YOLO, let’s test the working of YOLO v4. To test the darknet, first, we have to download a pre-trained model. The following model is trained for the MS COCO dataset.

Download YOLO v4 Model(https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view)
After downloading the _**yolov4.weights**_, copy to the darknet folder. Now make sure that you have the following files in the darknet folder.

#### Testing YOLO v4 in a single image

The command below is for running YOLO in a single image. 
Both of the commands mentioned below do the same functions. 
The first one is for detection from one image, 
the second one is for multiple use cases, for eg. 
detection from video and webcam.

```
Version #1

./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg

Version #2
 
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg -i 0 -thresh 0.25
```
 ## P.S. More info you can find in the full instruction.
 
## How to use batch_video_parser.py
0 Create env and install requirements.txt

i use pyenv so example how to create env based on pyenv

```
pyenv install -v 3.8.5
pyenv virtualenv 3.8.3 envname # create env
pyenv local envname# set autodetect local env(activate when cd to folder)

pip install -r requirements.txt
```

1 export all env variables(you should put your values)
```.env
export $(grep -v '^#' .env | xargs)
```
2 In file /darknet/cfg/coco.data parameter "names" should be 'path/to/folder/darknet/data/coco.names'

example:
```
names = /home/user/SpilnaSprava/Mollengo/YOLOBatchProcessing/darknet/data/coco.names
```

3 Start the script

```python
python batch_video_parser.py -v /home/user/SpilnaSprava/Mollengo/07_20200720135959_20200720145957.asf -b 8
```

-v videofile path
-b batch size (default 1)
-l logfile path /home/user/logfile.log

4 video_batching.py New version of vodeo batching

```python
python video_batching.py -v /home/ubuntu/achernov/07_20200720135959_20200720145957.asf
```

-v videofile path
all other inside the settings.py file

