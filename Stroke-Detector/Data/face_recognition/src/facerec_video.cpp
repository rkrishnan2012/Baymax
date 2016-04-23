/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

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

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        if (line.find(".DS_Store") != std::string::npos) {
            cout << "Ignoring .DS_Store file." << endl;
            continue;
        }
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            cout << path << endl;
            Mat image, image_resized;
            //  Read the image
            image = imread(path, 1);
            if (image.empty()) {
                 cerr << path << " could not be read!" << endl;
                 return;
            }
            Mat gray;
            //  Grayscale
            cvtColor(image, gray, CV_BGR2GRAY);
            //  Apply gaussian blur
            GaussianBlur(gray, gray, Size(5,5), 0, 0, BORDER_DEFAULT);
            /// Sobel edge detection
            Mat edges, grad_x, grad_y;
            Mat abs_grad_x, abs_grad_y;
            int scale = 1;
            int delta = 0;
            int ddepth = CV_16S;
            Sobel( gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_x, abs_grad_x );
            Sobel( gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_y, abs_grad_y );
            addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges );
            cv::resize(edges, image_resized, Size(200, 200));
            images.push_back(image_resized);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
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

void verifyAccuracy(BasicFaceRecognizer* model, vector<Mat> test_images, vector<int> test_labels){
    double truePositive = 0;  // Correctly found 1
    double trueNegative = 0;  // Correctly found 0
    double falsePositive = 0; // Predicted 1 but actual 0
    double falseNegative = 0; // Predicted 0 but actual 1

    // Shuffle the test images
    for (int k = 0; k < test_images.size(); k++) {
        int r = k + rand() % (test_images.size() - k); // careful here!
        swap(test_images[k], test_images[r]);
        swap(test_labels[k], test_labels[r]);
    }

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

    cout << "True pos:" << truePositive << endl;
    cout << "True neg:" << trueNegative << endl;
    cout << "False pos:" << falsePositive << endl;
    cout << "False neg:" << falseNegative << endl;
    
    double accuracy = (100 * (truePositive + trueNegative)) / test_labels.size();
    double TPR = (100 * (truePositive / (truePositive + falseNegative))); // Recall
    double FNR = (100 * (falseNegative / (truePositive + falseNegative)));
    double FPR = (100 * (falsePositive / (falsePositive + trueNegative)));
    double TNR = (100 * (trueNegative / (falsePositive + trueNegative)));
    double PPV = (100 * (truePositive / (truePositive + falsePositive))); // Precision
    double FDR = (100 * (falsePositive / (truePositive + falsePositive)));
    double FOR = (100 * (falseNegative / (falseNegative + trueNegative)));
    double NPV = (100 * (trueNegative / (falseNegative + trueNegative)));
    double f1Score = (2 * (PPV * TPR)) / (PPV + TPR);
    cout << "Accuracy of model is: " << accuracy <<
        "%" << endl;

    cout << "F1-Score of model is: " << f1Score <<
    "%" << endl;
}

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 5) {
        cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/training/csv.ext> </path/to/test/csv.ext> </path/to/device id>" << endl;
        cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/training/csv.ext> -- Path to the CSV file with the training face database." << endl;
        cout << "\t </path/to/test/csv.ext> -- Path to the CSV file with the test face database." << endl;
        cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
        exit(1);
    }
    // Get the path to your CSV:
    string fn_haar = string(argv[1]);
    string fn_training = string(argv[2]);
    string fn_test = string(argv[3]);
    int deviceId = atoi(argv[4]);
    // These vectors hold the images and corresponding labels:
    vector<Mat> training_images;
    vector<int> training_labels;

    vector<Mat> test_images;
    vector<int> test_labels;

    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        read_csv(fn_training, training_images, training_labels);
        read_csv(fn_test, test_images, test_labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening csv file . Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    // Get the height from the first image. We'll need this
    // later in code to reshape the training_images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = training_images[0].cols;
    int im_height = training_images[0].rows;
    // Create a FaceRecognizer and train it on the given training_images:
    Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
    cout << "Beginning training. " << endl;
    // ifstream f("fisher.yml");
    // if(f.good()){
    //   model->load("fisher.yml");
    // } else {
    model->train(training_images, training_labels);
    model->save("fisher.yml");
    // }
    cout << "Done training. " << endl;

    verifyAccuracy(model, test_images, test_labels);
    
    // Here is how to get the eigenvalues of this Eigenfaces model:
    Mat eigenvalues = model->getEigenValues();
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getEigenVectors();
    // Get the sample mean from the training data
    Mat mean = model->getMean();
    // Display or save the first, at most 16 Fisherfaces:
    for (int i = 0; i < min(16, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, im_height));
        // Show the image & apply a Bone colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
        // Display or save:
        imshow(format("fisherface_%d", i), cgrayscale);
    }
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    } else {
        cout << "Opened capture device ID " << deviceId << "." << endl;
    }
    // Holds the current frame from the Video device:
    Mat frame;
    for(;;) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(original, faces);

        // At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);
            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
            // verify this, by reading through the face recognition tutorial coming with OpenCV.
            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
            // input data really depends on the algorithm used.
            //
            // I strongly encourage you to play around with the algorithms. See which work best
            // in your scenario, LBPH should always be a contender for robust face recognition.
            //
            // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
            // face you have just found:
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height));
            // Now perform the prediction, see how easy that is:
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            // Create the text we will annotate the box with:
            string box_text = format("Prediction = %d", prediction);
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return 0;
}
