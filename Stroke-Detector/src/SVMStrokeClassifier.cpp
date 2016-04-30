/***************************************************************************************************************
😎 Bun $ g++ `pkg-config --cflags opencv` -o SVMClassifier SVMClassifier.cpp `pkg-config --libs opencv` -L/Users/Bunchhieng/opencv/3rdparty/ippicv/unpack/ippicv_osx/lib/
😎 Bun $ ./SVMClassifier ../Data/classifiers/face.xml ../Data/classifiers/mouth.xml ../Data/filtered-images/train/train.txt ../Data/filtered-images/test/test.txt ../Data/filtered-images/cv/cv.txt
***************************************************************************************************************/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace cv;
using namespace cv::face;
using namespace std;

const int PARAM_WIDTH = 50;
const int PARAM_HEIGHT = 50;

Mat preProcessImage(Mat& image, CascadeClassifier faceClassifier, CascadeClassifier mouthClassifier) {
    cvtColor(image, image, CV_RGB2GRAY);
    resize(image, image, Size(PARAM_WIDTH, PARAM_HEIGHT));
    image.convertTo(image, CV_32F);
    return image;
}

static void read_csv(const string &filename, vector<Mat> &images, vector<int> &labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    int neg = 0, pos = 0;
    while (getline(file, line)) {
        if (line.find(".DS_Store") != std::string::npos) {
            //  Ignore ds_store files, lol.
            continue;
        }
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        int type = (atoi(classlabel.c_str()) == 0); // reverse it since our csv is wrong
        if (type == 0) {
            if (neg > 700) {
                continue;
            } else {
                neg++;
            }
            neg++;
        } else {
            if (pos > 2000) {
                continue;
            } else {
                pos++;
            }
        }
        if (!path.empty() && !classlabel.empty()) {
            Mat image;
            //  Read the image
            image = imread(path, 1);
            if (image.empty()) {
                cerr << path << " could not be read!" << endl;
                return;
            }
            images.push_back(image);
            labels.push_back(type);
        }
    }
}

void preProcessImages(vector<Mat> &training_images, CascadeClassifier faceClassifier, CascadeClassifier mouthClassifier){
    for(int k = 0; k < training_images.size(); k++){
        preProcessImage(training_images[k], faceClassifier, mouthClassifier);
    }
}

void generateSingleMatrix(const vector<Mat>& training_images, Mat& training_mat){
    for(int k = 0; k < training_images.size(); k++){
        Mat orig = training_images[k];
        int ii = 0;
        for (int i = 0; i < orig.rows; i++) {
            for (int j = 0; j < orig.cols; j++) {
                float l = orig.at<float>(i, j);
                training_mat.at<float>(k, ii++) = l;
            }
        }
    }
}

double predictAll(Ptr<ml::SVM>& svm, Mat& test_mat, vector<int>& test_labels){
    double truePositive = 0;  // Correctly found 1
    double trueNegative = 0;  // Correctly found 0
    double falsePositive = 0; // Predicted 1 but actual 0
    double falseNegative = 0; // Predicted 0 but actual 1
    
    for(int i = 0; i < test_mat.rows; i++){
        int actual = test_labels[i];
        int prediction = svm->predict(test_mat.row(i));
        if(prediction == actual && prediction == 1){
            truePositive++;
        } else if(prediction == actual && prediction == 0){
            trueNegative++;
        } else {
            if(prediction == 1 && actual == 0){
                falsePositive++;
            } else if(prediction == 0 && actual == 1){
                falseNegative++;
            }
        }
    }

    cout << "   True pos:" << truePositive << endl;
    cout << "   True neg:" << trueNegative << endl;
    cout << "   False pos:" << falsePositive << endl;
    cout << "   False neg:" << falseNegative << endl;
    
    double accuracy = (100 * (truePositive + trueNegative)) / test_labels.size();
    double TPR = (100 * (truePositive / (truePositive + falseNegative))); // Recall
    if(truePositive + falseNegative == 0){
        TPR = 0;
    }
    //double FNR = (100 * (falseNegative / (truePositive + falseNegative)));
    //double FPR = (100 * (falsePositive / (falsePositive + trueNegative)));
    //double TNR = (100 * (trueNegative / (falsePositive + trueNegative)));
    double PPV = (100 * (truePositive / (truePositive + falsePositive))); // Precision
    if(truePositive + falsePositive == 0){
        PPV = 0;
    }
    
    cout << "   Precision:" << PPV << "%" << endl;
    cout << "   Recall:" << TPR << "%" << endl;
    
    //double FDR = (100 * (falsePositive / (truePositive + falsePositive)));
    //double FOR = (100 * (falseNegative / (falseNegative + trueNegative)));
    //double NPV = (100 * (trueNegative / (falseNegative + trueNegative)));
    double f1Score = (2 * (PPV * TPR)) / (PPV + TPR);
    if(PPV + TPR == 0){
        f1Score = 0;
    }
    cout << "   Accuracy of model is: " << accuracy <<
    "%" << endl;
    return f1Score;
}

int main(int argc, char **argv) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 6) {
        cout << "usage: " << argv[0] <<
        "</path/to/haar_face> </path/to/haar_mouth> </path/to/training/csv.ext> </path/to/cv/csv.ext> </path/to/test/csv.ext>" <<
        endl;
        cout << "\t </path/to/face_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/mouth_cascade> -- Path to the Haar Cascade for mouth detection." << endl;
        cout << "\t </path/to/training/csv.ext> -- Path to the CSV file with the training face database." << endl;
        cout << "\t </path/to/cv/csv.ext> -- Path to the CSV file with the cross validation face database." << endl;
        cout << "\t </path/to/test/csv.ext> -- Path to the CSV file with the test face database." << endl;
        exit(1);
    }
    
    
    

    // Get the path to your CSV:
    string fn_haar_face = string(argv[1]);
    string fn_haar_mouth = string(argv[2]);
    string fn_training = string(argv[3]);
    string fn_test = string(argv[4]);
    string fn_cv = string(argv[5]);

    CascadeClassifier faceClassifier;
    faceClassifier.load(fn_haar_face);

    CascadeClassifier mouthClassifier;
    mouthClassifier.load(fn_haar_mouth);

    vector<Mat> training_images;
    vector<int> training_labels;

    vector<Mat> cv_images;
    vector<int> cv_labels;

    vector<Mat> test_images;
    vector<int> test_labels;

    try {
        read_csv(fn_training, training_images, training_labels);
        read_csv(fn_cv, cv_images, cv_labels);
        read_csv(fn_test, test_images, test_labels);
    } catch (cv::Exception &e) {
        cerr << "Error opening csv file . Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    preProcessImages(training_images, faceClassifier, mouthClassifier);
    preProcessImages(test_images, faceClassifier, mouthClassifier);
    
    Mat training_mat((int)training_images.size(),PARAM_WIDTH * PARAM_HEIGHT, CV_32F, double(0));
    Mat test_mat((int)test_images.size(),PARAM_WIDTH * PARAM_HEIGHT, CV_32F, double(0));
    
    Mat labels((int)training_images.size(),1,CV_32S, (int*)training_labels.data());
    
    generateSingleMatrix(training_images, training_mat);
    generateSingleMatrix(test_images, test_mat);

    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setGamma(3);
    svm->train(training_mat, ml::ROW_SAMPLE, labels);

    cout << "Predicting" << endl;
    double fscore = predictAll(svm, test_mat, test_labels);
    cout << "F-score is: " << fscore << endl;
    cout << "Done!" << endl;
    
    return 0;
}
