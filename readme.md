Help come up with a new logo [here](https://www.hydraimageprocessor.com/logo-contest)

<img src=logo.png  width="300px" height="250px"/>

Hydra Image Processor (HIP)
===
Check out the website [here](https://www.hydraimageprocessor.com)

Most details will be found in the [wiki](https://github.com/ericwait/hydra-image-processor/wiki).

Hydra Image Processor is a hardware accelerated signal processing library written with [CUDA](https://developer.nvidia.com/cuda-zone). HIP aims to create a signal processing library that can be incorporated into many software tools. This library is licensed under BSD 3-Clause to encourage use in open-source and commercial software. My only plea is that if you find bugs or make changes that you contribute them back to this repository. Happy processing and enjoy!

# Quick Start Guide
## Windows
Requirements:
* Windows 7 or above
* MATLAB version R2016a and above (most heavily tested on R2017b)
* [CUDA capable device](https://developer.nvidia.com/cuda-gpus) with the latest drivers installed
* Watchdog registry values set (see [Registry Changes](#registry-changes) below)

### Registry Changes
_**IMPORTANT!!! PLEASE READ**_ Registry settings need to be set for HIP to run successfully on any appreciable data sizes. Please restart your computer after applying.

On Windows devices, the operating system is very particular about having immediate access and availability of the graphics card. With HIP operations that take more than a few hundred milliseconds, the operating system thinks that the graphics card is unresponsive to draw the screen. After a few attempts without success, Windows will attempt to restart the graphics driver. When this happens, HIP will crash because the hardware has been taken away mid-operation. There is a way to tell Windows to be more patient.

Included in the root of the repository are three TDR delay registry files. Each one contains setting to delay the operating system from reseting the graphics driver. I recommend using the ```tdrDelay_standard.reg``` file for most configurations. However, if you are using a older graphics card or a mobile version (like that in the Surface Book), I would recommend the ```tdrDelay_LONG.reg```. You could just use the LONG version no matter what your configuration. However, these settings will also delay legitimate errors with the graphics driver. Meaning that if you install a program that crashes the graphics driver, your computer might seem unresponsive for up to 15 minutes before the operating system steps in to recover.

Some of you might be rightfully weary of having your registry changed by some file you found on the Internet. I understand completely. Or maybe your just curious to know what each of the registry values mean. You can either search for "TDR Delay" with your favorite search engine or go to Microsoft's document page on TDR Delay [here](https://docs.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys). If you are still unsure, leave a comment in one of the [forums](https://www.hydraimageprocessor.com/forum).

### MATLAB Path
The quickest way to get HIP running is to use the library within MATLAB. First ensure that the appropriate registry values [have been set](#registry-changes) and that the computer has been restarted. The only thing left to do is ensure MATLAB can find the HIP package. HIP uses MATLAB's directory naming scheme to denote that it is a [package](https://www.mathworks.com/help/matlab/matlab_oop/scoping-classes-with-packages.html) (as opposed to a class or just a directory).

MATLAB needs to know of the location of the HIP package. This can be done a few ways. The location to the HIP package needs to be on MATLAB's path environment variable. This can be done by using the command ```addpath``` with the full path to the ```+HIP``` directory within the repository. This directory is found within this repository at ```/src/MATLAB/+HIP```. The addpath command is convenient because you can add it to the ```startup.m``` file that is loaded when [MATLAB starts](https://www.mathworks.com/help/matlab/ref/startup.html). The path to HIP can also be added using the menu within MATLAB. I don't recommend this method because it is harder to change if HIP is ever moved from where it was initially cloned. Lastly, the ```+HIP``` directory can be copied to the MATLAB directory within your users document directory. This is also not recommended because update to the repository will not be used unless this directory is copied again.

## Usage From MATLAB
Using HIP within MATLAB has been designed to be as easy as possible. Once the HIP package is on the [MATLAB path](#matlab-path), all you have to do is type ```HIP.``` and press the tab key after typing the period. This will list all of the functions currently available. Select the desired function, such as ```Gaussian``` (which will apply a Gaussian blurring). Then type the open parenthesis ```(``` and a list of input parameters will be presented. You can also use the ```F1``` key while your cursor is on the function name to get the full help description. 

Input parameters have been standardized and are typically:
* Input data (one to five dimensional array of any type)
* Explicit kernel or a three vector
* Function specific parameters
* Optional device number (pass an empty array ```[]``` to use all available devices)

There is typically only one output parameter which is an array the size and type (when possible) of the input array.

So for example to Gaussian blur an image would be:
```
imOut = HIP.Gaussian(im,[25,25,10],[],[]);
```

* ```imOut``` will be the same dimension and type of im
* ```im``` can have one to five dimensions and be of any type
* ```[25,25,20]``` are the sigma values (X,Y,Z) that define the Gaussian smoothing kernel. A zeros sigma indicates not to smooth in that direction.
* ```[]``` The first empty array indicates that this is an optional parameter. For Gaussian smoothing this parameter is to indicate how many times this smoothing should be performed.
* ```[]``` The last parameter is to indicate which GPU to use. Passing an empty array indicates that all devices can be utilized. HIP will distribute work across each device when appropriate.

## Python Bindings (alpha)
Requirements:
* Windows 7 or above
* [Visual Studio 2015 Community Edition](https://visualstudio.microsoft.com/vs/older-downloads)
* [Python (64-bit) 3.4 or greater](https://www.python.org/downloads)
* [Numpy 1.12 or greater](http://www.numpy.org)
* [CUDA capable device](https://developer.nvidia.com/cuda-gpus) with the latest drivers installed
* Watchdog registry values set (see [Registry Changes](#registry-changes))

Experimental Python bindings have been created for the Hydra Image Processing library, however, they must currently be built from source, since Python versions only support limited binary compatibility.

### Windows Build
The full Hydra Image Processing library can be built using the Visual Studio solution file ```src/c/CudaImageProccessing.sln```. Python bindings are built as part of the main solution, or can be built separately by ```src/c/CudaPy3DLL.vcxproj```.

Two environment variables must be defined for a successful build:
  1. PYTHON3_DIR - Root directory of the Python 3.x installation to build against
  2. NUMPY3_DIR - Root directory of the Numpy installation to build against

It is currently simplest to build Python bindings using the full solution file, at present, as the Python project is dependant on the core image processing library. NOTE: Unless you have ```Debug``` build of Python 3.x, the bindings must be built in ```Release``` mode.

### Installation and Usage
After a successful build the file ```src/Python/HIP.pyd``` should be placed in a directory on the PYTHON_PATH to make the Hydra Image Processing tools accessible from Python.

For use in Python import the HIP module:

```
import HIP
```

Exmple of blurring a random numpy volume:

```
import HIP
import numpy as np

im = np.random.randint(0,256, size=(512,512,25), dtype=np.uint8)
imOut = HIP.Gaussian(im, [25,25,10])
```

# Feedback
If you would like to provide feedback about this tutorial or HIP in general, please use the forum [here](https://www.hydraimageprocessor.com/forum).
