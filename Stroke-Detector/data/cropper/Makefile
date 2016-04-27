CPPFLAGS=`pkg-config --cflags opencv`
LDFLAGS=-g
LDLIBS=`pkg-config --libs opencv` -L$(HOME)/opencv/3rdparty/ippicv/unpack/ippicv_osx/lib/

all:	ImageCropper

ImageCropper:	ImageCropper.o
	g++ $(LDFLAGS) -o ImageCropper ImageCropper.o $(LDLIBS) 

ImageCropper.o:	ImageCropper.cpp
	g++ $(CPPFLAGS) -c ImageCropper.cpp

run: clean ImageCropper
	rm -rf filtered-images/
	mkdir -p filtered-images/palsy
	mkdir -p filtered-images/regular
	./ImageCropper ../original-images/palsy ../original-images/regular ../classifiers/face.xml ../classifiers/nose.xml ../classifiers/rightEye.xml
	./fileMover.sh
	rm -rf filtered-images
	rm -rf sorted/cv
	rm -rf sorted/test
	rm -rf sorted/train
	mv sorted/* ../filtered-images/
clean :	
	rm -f *.o