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

const int PARAM_WIDTH = 40;
const int PARAM_HEIGHT = 40;
double PARAM_SOBEL_SCALE = 3;
double PARAM_SOBEL_DELTA = 3;
double PARAM_GAMMA= 1; // 1.2
double PARAM_SIGMA= 1; // 1.2
double PARAM_THETA= 1.37445; // 1.2
double PARAM_LM= 1; // 1.2
double PARAM_PS= 1; // 1.2
double PARAM_TRAIN_GAMMA = 1.37445; // 1.2

void preProcessImage(const Mat& orig, Mat& dest, CascadeClassifier faceClassifier, CascadeClassifier mouthClassifier) {
    resize(dest, dest, Size(PARAM_WIDTH, PARAM_HEIGHT));
    Mat edges, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    int ddepth = CV_16S;
    Sobel( dest, grad_x, ddepth, 1, 0, 3, PARAM_SOBEL_SCALE, PARAM_SOBEL_DELTA, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    Sobel( dest, grad_y, ddepth, 0, 1, 3, PARAM_SOBEL_SCALE, PARAM_SOBEL_DELTA, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    //addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dest );
    
    int kernel_size = 10;
    double sig = PARAM_SIGMA, th = PARAM_THETA, lm = PARAM_LM, gm = PARAM_GAMMA, ps = PARAM_PS;
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
    //dest.convertTo(dest,CV_32F);
    //cv::filter2D(dest, dest, CV_32F, kernel);
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
            //  There are only about 256 palsy images
            pos++;
        }
        if (!path.empty() && !classlabel.empty()) {
            Mat image;
            //  Read the image
            image = imread(path, 0);
            if (image.empty()) {
                cerr << path << " could not be read!" << endl;
                return;
            }
            images.push_back(image);
            labels.push_back(type);
        }
    }
    cout << neg << " Regular faces." << endl;
    cout << pos << " Palsy faces." << endl;
}

void preProcessImages(const vector<Mat> &training_images, vector<Mat>& output_set,
                      CascadeClassifier faceClassifier, CascadeClassifier mouthClassifier){
    for(int k = 0; k < training_images.size(); k++){
        Mat newImage = training_images[k].clone();
        preProcessImage(training_images[k], newImage, faceClassifier, mouthClassifier);
        output_set.push_back(newImage);
    }
}

void generateSingleMatrix(const vector<Mat>& training_images, Mat& training_mat){
    for(int k = 0; k < training_images.size(); k++){
        Mat orig = training_images[k];
        Mat row = orig.reshape(1,1);
        row.copyTo(training_mat.row(k));
    }
    imshow("HI!", training_mat);
    waitKey();
}

double predictAll(Ptr<ml::SVM>& svm, Mat& test_mat, vector<int>& test_labels, bool print=false){
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
    
   
    
    //double FDR = (100 * (falsePositive / (truePositive + falsePositive)));
    //double FOR = (100 * (falseNegative / (falseNegative + trueNegative)));
    //double NPV = (100 * (trueNegative / (falseNegative + trueNegative)));
    double f1Score = (2 * (PPV * TPR)) / (PPV + TPR);
    if(PPV + TPR == 0){
        f1Score = 0;
    }
    
    if(print){
        cout << "   True pos:" << truePositive << endl;
        cout << "   True neg:" << trueNegative << endl;
        cout << "   False pos:" << falsePositive << endl;
        cout << "   False neg:" << falseNegative << endl;
        cout << "   Precision:" << PPV << "%" << endl;
        cout << "   Recall:" << TPR << "%" << endl;
        cout << "   Accuracy of model is: " << accuracy <<
        "%" << endl;
    }
    
    return f1Score;
}

Ptr<ml::SVM> trainSVM(const vector<Mat>& training_images, const vector<int>& training_labels,
                      CascadeClassifier faceClassifier, CascadeClassifier mouthClassifier){
    vector<Mat> training_processed;
    preProcessImages(training_images, training_processed, faceClassifier, mouthClassifier);
    
    Mat training_mat((int)training_images.size(),PARAM_WIDTH * PARAM_HEIGHT, training_processed[0].type(), double(0));
    
    Mat labels((int)training_images.size(),1,CV_32S, (int*)training_labels.data());
    
    
    generateSingleMatrix(training_processed, training_mat);
    
    Ptr<ml::SVM> svm;
    svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::POLY);
    svm->setGamma(3);
    svm->setDegree(3);
    svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001));
    //svm->train(training_mat, ml::ROW_SAMPLE, labels);
    svm->trainAuto(cv::ml::TrainData::create(training_mat,
                                             cv::ml::SampleTypes::ROW_SAMPLE,
                                             training_labels));
    

    return svm;
}

Ptr<ml::SVM> findBestModel(const vector<Mat>& training_images, const vector<int>& training_labels,
                           vector<Mat>& cv_images,  vector<int>& cv_labels,
                           CascadeClassifier faceClassifier, CascadeClassifier mouthClassifier){
    float best_theta = 1;
    float best_sigma = 1;
    float best_gamma = 1;
    float best_lm = 1;
    float best_ps = 1;
    
    double best_accuracy = 0;
    
    /*
     double PARAM_GAMMA= 1; // 1.2
     double PARAM_SIGMA= 1; // 1.2
     double PARAM_THETA= 1.37445; // 1.2
     */
    /*for(float p_theta = 0; p_theta < M_PI - M_PI/10; p_theta+=M_PI/16){
        for(float p_sigma = 0; p_sigma < 10; p_sigma+=1){
            for(float p_gamma = 0; p_gamma < 20; p_gamma+=5){
                for(float p_lm = 0; p_lm < 50; p_lm+=10){
                    for(float p_ps = 0; p_ps < 30; p_ps+=5){
                        PARAM_THETA = p_theta;
                        PARAM_SIGMA = p_sigma;
                        PARAM_GAMMA = p_gamma;
                        cout << "Trying theta=" << p_theta << ", sigma=" << p_sigma << ", gamma=" << p_gamma <<  ", lm=" << p_lm << ", ps=" << p_ps << endl;
                        
                        Ptr<ml::SVM> svm = trainSVM(training_images, training_labels, faceClassifier, mouthClassifier);
                        
                        vector<Mat> cv_processed;
                        preProcessImages(cv_images, cv_processed, faceClassifier, mouthClassifier);
                        
                        Mat cv_mat((int)cv_processed.size(),cv_processed[0].rows * cv_processed[0].cols, cv_processed[0].type(), double(0));
                        
                        generateSingleMatrix(cv_processed, cv_mat);
                        double accuracy = predictAll(svm, cv_mat, cv_labels);
                        if(accuracy > best_accuracy){
                            best_accuracy = accuracy;
                            best_theta = p_theta;
                            best_sigma = p_sigma;
                            best_gamma = p_gamma;
                            best_lm = p_lm;
                            best_ps = p_ps;
                            cout << "Accuracy: " << best_accuracy << endl;
                        }
                    }
                }
            }
        }
    }
    cout << "Done optimizing. Best values are: theta=" << best_theta << ", sigma=" << best_sigma << ", gamma=" << best_gamma <<  ", lm=" << best_lm << ", ps=" << best_ps << endl;
    PARAM_THETA = best_theta;
    PARAM_SIGMA = best_sigma;
    PARAM_GAMMA = best_gamma;
    PARAM_LM = best_lm;
    PARAM_PS = best_ps;*/
    
    Ptr<ml::SVM> svm = trainSVM(training_images, training_labels, faceClassifier, mouthClassifier);
    return svm;
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
        cout << "======TRAINING SET======" << endl;
        read_csv(fn_training, training_images, training_labels);
        cout << training_images.size() << " training images." << endl;
        cout << "======CV SET======" << endl;
        read_csv(fn_cv, cv_images, cv_labels);
        cout << cv_images.size() << " cv images." << endl;
        cout << "======TEST SET======" << endl;
        read_csv(fn_test, test_images, test_labels);
        cout << test_images.size() << " test images." << endl;
    } catch (cv::Exception &e) {
        cerr << "Error opening csv file . Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    
    Ptr<ml::SVM> svm = findBestModel(training_images, training_labels, cv_images, cv_labels, faceClassifier, mouthClassifier);

    vector<Mat> training_processed;
    vector<Mat> test_processed;
    preProcessImages(training_images, training_processed, faceClassifier, mouthClassifier);
    preProcessImages(test_images, test_processed, faceClassifier, mouthClassifier);
    
    Mat training_mat((int)training_images.size(),training_processed[0].rows * training_processed[0].cols, training_processed[0].type(), double(0));
    Mat test_mat((int)test_images.size(),test_processed[0].rows * test_processed[0].cols, test_processed[0].type(), double(0));
    
    Mat labels((int)training_images.size(),1,CV_32S, (int*)training_labels.data());
    
    generateSingleMatrix(training_processed, training_mat);
    generateSingleMatrix(test_processed, test_mat);
    
    double fscore = predictAll(svm, test_mat, cv_labels, true);
    cout << "Final F-score is: " << fscore << endl;
    
    Mat viz;
    training_mat.convertTo(viz,CV_8U,1);
    applyColorMap(viz, viz, COLORMAP_JET);
    imshow("Training matrix", viz);
    waitKey();

    return 0;
}
