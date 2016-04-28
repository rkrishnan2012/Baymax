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

// This will be automatically changed using gradient descent.
double PARAM_GAUSS_X = 1; // 17
double PARAM_GAUSS_Y = 1; // 17
double PARAM_SOBEL_SCALE = 3;
double PARAM_SOBEL_DELTA = 3;
double PARAM_WIDTH = 50;
double PARAM_HEIGHT = 50;
double PARAM_GAMMA = 1; // 1.2
double PARAM_SIGMA = 1.5; // 1.2
double PARAM_THETA = 1.76715; // 1.2

Mat preProcessImage(Mat orig, CascadeClassifier faceClassifier, CascadeClassifier mouthClassifier) {
    Mat image = orig.clone();
    //  Find the mouth
    vector<Rect_<int> > mouths;
    mouthClassifier.detectMultiScale(image, mouths);
    if (mouths.size() == 0) {
        //cerr << "There isn't a mouth in a picture." << endl;
        //imshow("full", image);
        //waitKey();
    }
    // Rect mouthRect = mouths[0];
    //  Crop the pic to the mouth
    Rect myROI(0, image.size().height / 2, image.size().width, image.size().height / 2);
    //image = image(myROI);
    //  Resize it to 200x200
    //cv::resize(image, image, Size(PARAM_WIDTH, PARAM_HEIGHT));
    //  Grayscale
    cvtColor(image, image, CV_RGB2GRAY);
    //  Apply gaussian blur
    //GaussianBlur(image, image, Size(PARAM_GAUSS_X,PARAM_GAUSS_Y), 0, 0, BORDER_DEFAULT);
    /// Apply Histogram Equalization
    //equalizeHist(image, image);
    /// Sobel edge detection
    Mat edges, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    int scale = PARAM_SOBEL_SCALE;
    int delta = PARAM_SOBEL_DELTA;
    int ddepth = CV_16S;
    Sobel(image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    Sobel(image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    //addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image );
    image.convertTo(image, CV_32F);
    int kernel_size = 10;
    double sig = PARAM_SIGMA, th = PARAM_THETA, lm = 1.0, gm = PARAM_GAMMA, ps = 0;
    Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, lm, gm, ps);
    filter2D(image, image, CV_32F, kernel);
    Mat viz;
    image.convertTo(viz, CV_8U, 1.0 / 255.0);     // move to proper[0..255] range to show it
    imshow("k", kernel);
    imshow("d", viz);
    //waitKey();
    //  Resize to 200x200 size
    resize(image, image, Size(PARAM_WIDTH, PARAM_HEIGHT));
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
            if (neg > 100) {
                continue;
            } else {
                neg++;
            }
        } else {
            if (pos > 1000) {
                continue;
            } else {
                pos++;
            }
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
    cout << neg << " negatives." << endl;
    cout << pos << " positives." << endl;
}

Mat normalize_to_1D(String imgname) {
    Mat img_mat = imread(imgname, 0); // I used 0 for greyscale
    Mat training_mat(1, img_mat.rows * img_mat.cols, CV_32FC1);
    int file_num = 0;
    int ii = 0; // Current column in training_mat
    for (int i = 0; i < img_mat.rows; i++) {
        for (int j = 0; j < img_mat.cols; j++) {
            training_mat.at<float>(file_num, ii++) = img_mat.at<uchar>(i, j);
        }
    }
    return training_mat;
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

    Mat training_image;
    // training_image = normalize_to_1D(
    //"/Users/Bunchhieng/Documents/Bunchhieng/HiveLabs/Baymax/Stroke-Detector/Data/filtered-images/train/palsy/000113.jpg");
    int labels[1] = {1};
    Mat labelsMat(1, 1, CV_32S, labels);

    // These vectors hold the images and corresponding labels:
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

    // CascadeClassifier xml files
    CascadeClassifier mouthClassifier;
    mouthClassifier.load(fn_haar_mouth);

    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar_face);

    for (int i = 0; i < test_images.size(); i++) {
        //  Preprocess the image using the parameters
        test_images[i] = preProcessImage(test_images.at(i), haar_cascade, mouthClassifier);
    }
    // // Store all 1D image matrix
    // Mat_<float> trainingDataMat;
    // // 1D label matrix
    // Mat_<float> labelsMat;
    // // Data for visual representation
    // int width = 512, height = 512;
    // Mat image = Mat::zeros(height, width, CV_8UC3);
    //
    // // Set up training data
    // float labels[4] = {1.0, -1.0, -1.0, -1.0};
    // Mat labelsMat(4, 1, CV_32FC1, labels);
    //
    // float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    // Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    //
    // Set up SVM's parameters
    // CvSVMParams params;
    // params.svm_type    = CvSVM::C_SVC;
    // params.kernel_type = CvSVM::LINEAR;
    // params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Convert vector to matrix
    Mat T = Mat(training_images.size(), 1, CV_32FC1);
    memcpy(T.data, training_images.data(), training_images.size() * sizeof(float));
    Mat L = Mat(training_labels.size(), 1, CV_32FC1);
    memcpy(L.data, training_labels.data(), training_labels.size() * sizeof(float));

    cout << T << endl;
    Ptr<ml::SVM> svm = ml::SVM::create();
    // edit: the params struct got removed,
    // we use setter/getter now:
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setGamma(3);
    svm->train(T, ml::ROW_SAMPLE, L);
    float predictVal = svm->predict(training_image);
    cout << predictVal << endl;
    return 0;
}
