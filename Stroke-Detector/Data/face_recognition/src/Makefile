CPPFLAGS=`pkg-config --cflags opencv`
LDFLAGS=-g
LDLIBS=`pkg-config --libs opencv` -L$(HOME)/opencv/3rdparty/ippicv/unpack/ippicv_osx/lib/

all:	facerec_video

facerec_video:	facerec_video.o
	g++ $(LDFLAGS) -o facerec_video facerec_video.o $(LDLIBS) 

facerec_video.o:	facerec_video.cpp
	g++ $(CPPFLAGS) -c facerec_video.cpp

run: clean facerec_video
	python ../data/create_csv.py ../data/train > ../data/train/train.txt
	python ../data/create_csv.py ../data/test > ../data/test/test.txt
	python ../data/create_csv.py ../data/cv > ../data/cv/cv.txt
	./facerec_video Face.xml parojos.xml ../data/train/train.txt ../data/cv/cv.txt ../data/test/test.txt 0

clean :	
	rm -f *.o