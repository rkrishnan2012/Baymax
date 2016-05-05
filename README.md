# Project Baymax Stroke-Detector
Baymax is an autonomous drone equipped with a camera to detect if a person is suffering a stroke. 

This project uses FisherFaces in OpenCV to determine whether a given face is normal or has Bell's Facial Palsy.

<p align="center">
  <img src="http://images6.fanpop.com/image/photos/37600000/Transparent-Baymax-big-hero-6-37653146-415-500.png" width="100"/>
</p>

### Project Structure
Drone - The Arduino code to be run on the drone
PhoneApp - Android code to run on the phone connected to the drone
Stroke-Detector -The face recognition and stroke detection algorithm
    
### Version
1.0.0

### Tech

The Stroke-Detector uses the following technology to function:
* [OpenCV] - Computer vision framework for C++
* Fisher Discriminant Analysis (to form a decision boundary)

### Getting Started
* Install Brew, OpenCV, and clone the project as shown below
* Download tons of images of regular and palsy faces and place them in ```Stroke-Detector/data/original-images``` (we have provided about 2k for you)
* ```cd Stroke-Detector/data/cropper && make run``` - This will crop, filter, and sort all the images in random order and divide it in a 60/20/20 split for Training, Test, and Cross-Validation.
* ```cd Stroke-Detector/src && make run``` - This will run the detection and output the results.


### Installation (Mac 10.9+)

Install Brew:
```sh
$ cd ~
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ brew update
```

Install Pre-requisites:
```sh
$ brew install python3
$ brew install linkapps
$ brew install cmake pkg-config
$ brew install jpeg libpng libtiff openexr
$ brew install eigen tbb
```


Install OpenCV:
```sh
$ cd ~
$ git clone https://github.com/Itseez/opencv.git
$ git clone https://github.com/Itseez/opencv_contrib
$ cd ~/opencv_contrib
$ git checkout 3.1.0
$ cd ~/opencv
$ git checkout 3.1.0
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D WITH_IPP=ON \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D PYTHON3_PACKAGES_PATH=~/.virtualenvs/cv3/lib/python3.5/site-packages \
	-D PYTHON3_LIBRARY=/usr/local/Cellar/python3/3.5.1/Frameworks/Python.framework/Versions/3.5/lib/libpython3.5m.dylib \
	-D PYTHON3_INCLUDE_DIR=/usr/local/Cellar/python3/3.5.1/Frameworks/Python.framework/Versions/3.5/include/python3.5m \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D BUILD_EXAMPLES=ON \
	-D BUILD_opencv_python3=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules ..
$ make -j4
$ sudo make uninstall # Remove old versions of Opencv3 and junk.
$ sudo make install
```

Get the source for this project: 
```sh
$ cd ~
$ git clone [git-repo-url] baymax-face-recognizer
```


Compile everything:
```sh
$ cd ~/baymax-face-recognizer/data
$ python create_csv.py $(pwd) > filepath.txt
$ cd ~/baymax-face-recognizer/src
$ g++ `pkg-config --cflags opencv` -o facerec_video facerec_video.cpp `pkg-config --libs opencv` -L/Users/rkrishnan/opencv/3rdparty/ippicv/unpack/ippicv_osx/lib/  
```


### Errors
Here are some errors you run into sometimes if you are running on mac:
###### err: dyld: Symbol not found: __cg_jpeg_resync_to_restart
```sh
$ cd /usr/local/lib
$ rm libgif.dylib
$ ln -s /System/Library/Frameworks/ImageIO.framework/Resources/libGIF.dylib libGIF.dylib
$ rm libjpeg.dylib
$ ln -s /System/Library/Frameworks/ImageIO.framework/Resources/libJPEG.dylib libJPEG.dylib
$ rm libtiff.dylib
$ ln -s /System/Library/Frameworks/ImageIO.framework/Resources/libTIFF.dylib libTIFF.dylib
$ rm libpng.dylib
$ ln -s /System/Library/Frameworks/ImageIO.framework/Resources/libPng.dylib libPng.dylib
```
 
License
----
MIT