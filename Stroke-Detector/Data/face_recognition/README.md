# Project Baymax Face-Recognizer
This is a sample project that uses FisherFaces in OpenCV to detect faces within images that are fed directly from the webcam.

<p align="center">
  <img src="http://images6.fanpop.com/image/photos/37600000/Transparent-Baymax-big-hero-6-37653146-415-500.png" width="100"/>
</p>


### Version
1.0.0

### Tech

Face-Recognizer uses the following software to function:

* [OpenCV] - Computer vision framework for C++

### Installation (For OSX 10.11)

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

Run it (Training the set will take forever, so grab some coffee.):
```sh
$ ./facerec_video haarcascade_frontalface_default.xml ../data/filepath.txt 0
```

### Errors (some errors / solutions)

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
 


### To-dos

 - Try with the Yale face dataset
 - Crop/Resize the training images the proper way (not just 100x100 random crop)


License
----
MIT

