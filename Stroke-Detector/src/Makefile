CPPFLAGS=`pkg-config --cflags opencv`
LDFLAGS=-g
LDLIBS=`pkg-config --libs opencv` -L$(HOME)/opencv/3rdparty/ippicv/unpack/ippicv_osx/lib/

all: SVMStrokeClassifier StrokeClassifier

StrokeClassifier:	StrokeClassifier.o
	g++ $(LDFLAGS) -o StrokeClassifier StrokeClassifier.o $(LDLIBS) 

SVMStrokeClassifier:	SVMStrokeClassifier.o
	g++ $(LDFLAGS) -o SVMStrokeClassifier SVMStrokeClassifier.o $(LDLIBS) 

StrokeClassifier.o:	StrokeClassifier.cpp
	g++ $(CPPFLAGS) -c StrokeClassifier.cpp

SVMStrokeClassifier.o:	SVMStrokeClassifier.cpp
	g++ $(CPPFLAGS) -c SVMStrokeClassifier.cpp

run: clean SVMStrokeClassifier StrokeClassifier
	#python ../data/filtered-images/create_csv.py ../data/filtered-images/train > ../data/filtered-images/train/train.txt
	#python ../data/filtered-images/create_csv.py ../data/filtered-images/test > ../data/filtered-images/test/test.txt
	#python ../data/filtered-images/create_csv.py ../data/filtered-images/cv > ../data/filtered-images/cv/cv.txt
	./StrokeClassifier ../data/classifiers/face.xml ../data/classifiers/nose.xml ../data/filtered-images/train/train.txt ../data/filtered-images/cv/cv.txt ../data/filtered-images/test/test.txt 0
	#./SVMStrokeClassifier ../data/classifiers/face.xml ../data/classifiers/nose.xml ../data/filtered-images/train/train.txt ../data/filtered-images/cv/cv.txt ../data/filtered-images/test/test.txt

clean :	
	rm -f *.o