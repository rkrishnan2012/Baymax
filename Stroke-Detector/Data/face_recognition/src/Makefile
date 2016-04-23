CPPFLAGS=`pkg-config --cflags opencv`
LDFLAGS=-g
LDLIBS=`pkg-config --libs opencv` -L$(HOME)/opencv/3rdparty/ippicv/unpack/ippicv_osx/lib/

all:	facerec_video

facerec_video:	facerec_video.o
	g++ $(LDFLAGS) -o facerec_video facerec_video.o $(LDLIBS) 

facerec_video.o:	facerec_video.cpp
	g++ $(CPPFLAGS) -c facerec_video.cpp

run: facerec_video
	python ../data/create_csv.py ../data/train > ../data/train/train.txt
	python ../data/create_csv.py ../data/test > ../data/train/test.txt
	./facerec_video haarcascade_frontalface_default.xml ../data/train/train.txt 0

clean :	
	rm *.o