#include "opencv2/core.hpp"
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
double PARAM_HEIGHT= 50;
double PARAM_GAMMA= 1; // 1.2
double PARAM_SIGMA= 1; // 1.2
double PARAM_THETA= 1.37445; // 1.2

Mat preProcessImage(Mat orig, CascadeClassifier faceClassifier, CascadeClassifier mouthClassifier){
    Mat image=orig.clone();
  
    cvtColor(image, image, CV_BGR2GRAY);

    image.convertTo(image,CV_32F);
    int kernel_size = 10;
    double sig = PARAM_SIGMA, th = PARAM_THETA, lm = 1.0, gm = PARAM_GAMMA, ps = 0;
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
    cv::filter2D(image, image, CV_32F, kernel);
    Mat viz;
    image.convertTo(viz,CV_8U,20.0/255.0);     // move to proper[0..255] range to show it
    imshow("k",kernel);
    applyColorMap(viz, viz, COLORMAP_BONE);
    imshow("gabor",viz);
    //waitKey();
    
    //  Resize to 200x200 size
    cv::resize(image, image, Size(PARAM_WIDTH, PARAM_HEIGHT));
    return image;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
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
        if(type == 0){
            if(neg > 200){
                continue;
            } else {
                neg++;
            }
        } else {
            if(pos > 1000){
                continue;
            } else {
                pos++;
            }
        }
        if(!path.empty() && !classlabel.empty()) {
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
    cout << neg << " negatives." << endl;
    cout << pos << " positives." << endl;
}

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

double verifyAccuracy(BasicFaceRecognizer* model, vector<Mat> test_images, vector<int> test_labels){
    double truePositive = 0;  // Correctly found 1
    double trueNegative = 0;  // Correctly found 0
    double falsePositive = 0; // Predicted 1 but actual 0
    double falseNegative = 0; // Predicted 0 but actual 1

    // Shuffle the test images
    /*for (int k = 0; k < test_images.size(); k++) {
        int r = k + rand() % (test_images.size() - k); // careful here!
        swap(test_images[k], test_images[r]);
        swap(test_labels[k], test_labels[r]);
    }*/
    cout << "Verifying against " << test_images.size() << " test images." << endl;
    for(int i = 0; i < test_images.size(); i++){
        int actual = test_labels.at(i);
        int prediction = model->predict(test_images.at(i));
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
    double FNR = (100 * (falseNegative / (truePositive + falseNegative)));
    double FPR = (100 * (falsePositive / (falsePositive + trueNegative)));
    double TNR = (100 * (trueNegative / (falsePositive + trueNegative)));
    double PPV = (100 * (truePositive / (truePositive + falsePositive))); // Precision
    if(truePositive + falsePositive == 0){
        PPV = 0;
    }

    cout << "   Precision:" << PPV << "%" << endl;
    cout << "   Recall:" << TPR << "%" << endl;

    double FDR = (100 * (falsePositive / (truePositive + falsePositive)));
    double FOR = (100 * (falseNegative / (falseNegative + trueNegative)));
    double NPV = (100 * (trueNegative / (falseNegative + trueNegative)));
    double f1Score = (2 * (PPV * TPR)) / (PPV + TPR);
    if(PPV + TPR == 0){
        f1Score = 0;
    }
    cout << "   Accuracy of model is: " << accuracy <<
        "%" << endl;
    return f1Score;
}

Ptr<BasicFaceRecognizer> findBestModel(CascadeClassifier faceClassifier,
        CascadeClassifier mouthClassifier,
        vector<Mat> training_images, vector<int> training_labels,
        vector<Mat> cv_images, vector<int> cv_labels){
    //  Optimize for the PARAM_GAUSS parameter
    double best_accuracy = 0;
    /*float best_theta = 1;
    for(float p_theta = 0; p_theta < M_PI - M_PI/10; p_theta+=M_PI/16){
        Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
        PARAM_THETA = p_theta;
        vector<Mat> temp_processed_images;
        vector<int> temp_processed_labels;
        vector<Mat> temp_cv_images;
        vector<int> temp_cv_labels;
        for(int i = 0; i < training_images.size(); i++){
            //  Preprocess the test image using the parameters
            temp_processed_images.push_back(preProcessImage(training_images.at(i), faceClassifier, mouthClassifier));
            temp_processed_labels.push_back(training_labels.at(i));
        }
        for(int i = 0; i < cv_images.size(); i++){
            //  Preprocess the cross validation image using the parameters
            temp_cv_images.push_back(preProcessImage(cv_images.at(i), faceClassifier, mouthClassifier));
            temp_cv_labels.push_back(cv_labels.at(i));
        }
        model->train(temp_processed_images, temp_processed_labels);
        double accuracy = verifyAccuracy(model, temp_cv_images, temp_cv_labels);
        if(accuracy > best_accuracy){
            best_accuracy = accuracy;
            best_theta = p_theta;
            cout << "Accuracy: " << best_accuracy << endl;
        }
    }
    cout << "Done optimizing. Best p_theta value is " << best_theta << endl;
    PARAM_THETA = best_theta;
   float param_sigma = 1;
    for(float p_sigma = 0; p_sigma < 2; p_sigma+=.3){
        Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
        PARAM_SIGMA = p_sigma;
        vector<Mat> temp_processed_images;
        vector<int> temp_processed_labels;
        vector<Mat> temp_cv_images;
        vector<int> temp_cv_labels;
        for(int i = 0; i < training_images.size(); i++){
            //  Preprocess the test image using the parameters
            temp_processed_images.push_back(preProcessImage(training_images.at(i), faceClassifier, mouthClassifier));
            temp_processed_labels.push_back(training_labels.at(i));
        }
        for(int i = 0; i < cv_images.size(); i++){
            //  Preprocess the cross validation image using the parameters
            temp_cv_images.push_back(preProcessImage(cv_images.at(i), faceClassifier, mouthClassifier));
            temp_cv_labels.push_back(cv_labels.at(i));
        }
        model->train(temp_processed_images, temp_processed_labels);
        double accuracy = verifyAccuracy(model, temp_cv_images, temp_cv_labels);
        if(accuracy > best_accuracy){
            best_accuracy = accuracy;
            param_sigma = p_sigma;
            cout << "Best f-score: " << best_accuracy << endl;
        }
    }
    cout << "Done optimizing. Best p_sigma value is " << param_sigma << endl;
    PARAM_SIGMA = param_sigma;*/
    /*float best_gamma = 1;
    for(float p_gamma = .9; p_gamma < 2; p_gamma+=.1){
        Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
        PARAM_GAMMA = p_gamma;
        vector<Mat> temp_processed_images;
        vector<int> temp_processed_labels;
        vector<Mat> temp_cv_images;
        vector<int> temp_cv_labels;
        for(int i = 0; i < training_images.size(); i++){
            //  Preprocess the test image using the parameters
            temp_processed_images.push_back(preProcessImage(training_images.at(i), faceClassifier, mouthClassifier));
            temp_processed_labels.push_back(training_labels.at(i));
        }
        for(int i = 0; i < cv_images.size(); i++){
            //  Preprocess the cross validation image using the parameters
            temp_cv_images.push_back(preProcessImage(cv_images.at(i), faceClassifier, mouthClassifier));
            temp_cv_labels.push_back(cv_labels.at(i));
        }
        model->train(temp_processed_images, temp_processed_labels);
        double accuracy = verifyAccuracy(model, temp_cv_images, temp_cv_labels);
        if(accuracy > best_accuracy){
            best_accuracy = accuracy;
            best_gamma = p_gamma;
            cout << "Accuracy: " << best_accuracy << endl;
        }
    }
    cout << "Done optimizing. Best p_gamma value is " << best_gamma << endl;
    PARAM_GAMMA = best_gamma;*/
    /*int best_p_gauss = 1;
    for(int p_gauss = 1; p_gauss < 50; p_gauss+=2){
        Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
        PARAM_GAUSS_X = p_gauss;
        PARAM_GAUSS_Y = p_gauss;
        vector<Mat> temp_processed_images;
        vector<int> temp_processed_labels;
        vector<Mat> temp_cv_images;
        vector<int> temp_cv_labels;
        for(int i = 0; i < training_images.size(); i++){
            //  Preprocess the test image using the parameters
            temp_processed_images.push_back(preProcessImage(training_images.at(i), mouthClassifier));
            temp_processed_labels.push_back(training_labels.at(i));
        }
        for(int i = 0; i < cv_images.size(); i++){
            //  Preprocess the cross validation image using the parameters
            temp_cv_images.push_back(preProcessImage(cv_images.at(i), mouthClassifier));
            temp_cv_labels.push_back(cv_labels.at(i));
        }
        model->train(temp_processed_images, temp_processed_labels);
        double accuracy = verifyAccuracy(model, temp_cv_images, temp_cv_labels);
        if(accuracy > best_accuracy){
            best_accuracy = accuracy;
            best_p_gauss = p_gauss;
            cout << "Accuracy: " << best_accuracy << endl;
        }
    }
    cout << "Done optimizing. Best p_gauss value is " << best_p_gauss << endl;
    PARAM_GAUSS_X = best_p_gauss;
    PARAM_GAUSS_Y = best_p_gauss;
    //  Optimize for the PARAM_SOBEL parameter
    int best_p_sobel = 1;
    for(int p_sobel = 1; p_sobel < 10; p_sobel+=.5){
        Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
        PARAM_SOBEL_DELTA = p_sobel;
        vector<Mat> temp_processed_images;
        vector<int> temp_processed_labels;
        vector<Mat> temp_cv_images;
        vector<int> temp_cv_labels;
        for(int i = 0; i < training_images.size(); i++){
            //  Preprocess the test image using the parameters
            temp_processed_images.push_back(preProcessImage(training_images.at(i), mouthClassifier));
            temp_processed_labels.push_back(training_labels.at(i));
        }
        for(int i = 0; i < cv_images.size(); i++){
            //  Preprocess the cross validation image using the parameters
            temp_cv_images.push_back(preProcessImage(cv_images.at(i), mouthClassifier));
            temp_cv_labels.push_back(cv_labels.at(i));
        }
        model->train(temp_processed_images, temp_processed_labels);
        double accuracy = verifyAccuracy(model, temp_cv_images, temp_cv_labels);
        if(accuracy > best_accuracy){
            best_accuracy = accuracy;
            best_p_sobel = p_sobel;
            cout << "Accuracy: " << best_accuracy << endl;
        }
    }
    cout << "Done optimizing. Best p_sobel value is " << best_p_sobel << endl;
    PARAM_SOBEL_DELTA = best_p_sobel;*/
    Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
    cout << "Pre-processing images." << endl;
    for(int i = 0; i < training_images.size(); i++){
        //  Preprocess the image using the parameters
        cout << ((100 * i) / training_images.size()) << "% done. Image #" << i << endl;
        training_images[i] = preProcessImage(training_images.at(i), faceClassifier, mouthClassifier);
    }
    cout << "Starting training." << endl;
    model->train(training_images, training_labels);
    //model->load("fisher.yml");
    cout << "Done training final model." << endl;
    return model;
}

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 7) {
        cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/training/csv.ext> </path/to/test/csv.ext> </path/to/device id>" << endl;
        cout << "\t </path/to/face_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/mouth_cascade> -- Path to the Haar Cascade for mouth detection." << endl;
        cout << "\t </path/to/training/csv.ext> -- Path to the CSV file with the training face database." << endl;
        cout << "\t </path/to/cv/csv.ext> -- Path to the CSV file with the cross validation face database." << endl;
        cout << "\t </path/to/test/csv.ext> -- Path to the CSV file with the test face database." << endl;
        cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
        exit(1);
    }
    // Get the path to your CSV:
    string fn_haar_face = string(argv[1]);
    string fn_haar_mouth = string(argv[2]);
    string fn_training = string(argv[3]);
    string fn_cv = string(argv[4]);
    string fn_test = string(argv[5]);
    int deviceId = atoi(argv[6]);

    // These vectors hold the images and corresponding labels:
    vector<Mat> training_images;
    vector<int> training_labels;

    vector<Mat> cv_images;
    vector<int> cv_labels;

    vector<Mat> test_images;
    vector<int> test_labels;

    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        read_csv(fn_training, training_images, training_labels);
        read_csv(fn_cv, cv_images, cv_labels);
        read_csv(fn_test, test_images, test_labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening csv file . Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    CascadeClassifier mouthClassifier;
    mouthClassifier.load(fn_haar_mouth);

    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar_face);

    Ptr<BasicFaceRecognizer> model =  findBestModel(haar_cascade, mouthClassifier, training_images, training_labels, cv_images, cv_labels);
    for(int i = 0; i < test_images.size(); i++){
        //  Preprocess the image using the parameters
        test_images[i] = preProcessImage(test_images.at(i), haar_cascade, mouthClassifier);
    }
    double accuracy = verifyAccuracy(model, test_images, test_labels);
    cout << "F1-Score of model is: " << accuracy <<
    "%" << endl;

    // Here is how to get the eigenvalues of this Eigenfaces model:
    Mat eigenvalues = model->getEigenValues();
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getEigenVectors();
    // Get the sample mean from the training data
    Mat mean = model->getMean();

    //
    // Get the height from the first image. We'll need this
    // later in code to reshape the training_images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = training_images[0].cols;
    int im_height = training_images[0].rows;

    // Display or save the Eigenfaces:
    for (int i = 0; i < W.cols; i++) {
        string msg = format("Eigenvalue #%d = %.10f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, test_images[0].rows));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
        cv::resize(cgrayscale, cgrayscale, Size(500, 500));
        imshow(format("eigenface_%d", i), cgrayscale);
        waitKey();
    }

    return 0;
}
