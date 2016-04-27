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
#include <dirent.h>
#include <limits.h>
#include <algorithm>

using namespace cv;
using namespace cv::face;
using namespace std;

CascadeClassifier faceClassifier;
CascadeClassifier noseClassifier;
CascadeClassifier rightEyeClassifier;


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            Mat image;
            image = imread(path, 0);
            if (image.empty()) {
                 cerr << path << " could not be read!" << endl;
                 return;
            }
            images.push_back(image);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void cropImage(char* path, char* outputPath){
    Mat image;
    image = imread(path, 1);
    if(!image.empty()){
        vector< Rect_<int> > faces;
        Mat face_resized;
        int im_width = 200;
        int im_height = 200;
        // Find the face
        faceClassifier.detectMultiScale(image, faces);
        for(int i = 0; i < faces.size() && i < 2; i++) {
          Rect face_i = faces[i];
          Mat cropped;

          cout << "ORIG:" << image.size().width << ", " << image.size().height << endl;
          cout << face_i.width << ", " << face_i.height << endl;
          cout << face_i.x << ", " << face_i.y << endl;
          // Crop the face
          cropped = image(face_i);

          vector< Rect_<int> > nose;
          noseClassifier.detectMultiScale(cropped, nose);
          if(nose.size() != 0){
              //  rectangle(cropped, nose[0], CV_RGB(255, 0,0), 1);              
          }

          vector< Rect_<int> > rEye;
          rightEyeClassifier.detectMultiScale(cropped, rEye);
          if(rEye.size() != 0){
               // rectangle(cropped, rEye[0], CV_RGB(0, 255,0), 1);              
          }

          if(nose.size() == 0 || rEye.size() == 0){
            return;
          }

          if(rEye[0].y + rEye[0].height < nose[0].y + nose[0].height) return;
          
          // Increase the rectangle Size
          cv::Size deltaSize(face_i.width * .3, 0); // 0.1f = 10/100
          cv::Point offset( deltaSize.width/2, deltaSize.height/2);
          face_i -= deltaSize;
          face_i += offset;
          cropped = image(face_i);

          
          // Resize the face to 200x200
          cv::resize(cropped, face_resized, Size(im_width, im_height));
          vector<int> compression_params;
          compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
          compression_params.push_back(3);
          imwrite(outputPath, face_resized, compression_params);
          cout << "Saved image to " << outputPath << endl;
          //rectangle(image, face_i, CV_RGB(0, 255,0), 1);
          // imshow("images", image);
          // waitKey();
        }
    }
}

void read_files(const char* path, string type){
    DIR *dirp = opendir(path);
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL){
        char buffer [256], outPath[256];
        snprintf (buffer, 256, "%s/%s", path, dp->d_name);
        snprintf (outPath, 256, "%s/%s/%s", "filtered-images",
          type.c_str(), dp->d_name);
        cropImage(buffer, outPath);
    }
    (void)closedir(dirp);
}

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 6) {
        cout << "usage: " << argv[0] << " </path/to/palsy/folder> </path/to/regular/folder> </path/to/haar_cascade>" << endl;
        cout << "\t </path/to/palsy/folder> -- Path to the Bell's palsy images directory downloaded." << endl;
        cout << "\t </path/to/regular/folder> -- Path to the Regular images directory downloaded." << endl;
        cout << "\t </path/to/haar_cascade> -- Path to HAAR facial classifier." << endl;
        cout << "\t </path/to/haar_cascade> -- Path to HAAR leftEye classifier." << endl;
        cout << "\t </path/to/haar_cascade> -- Path to HAAR rightEye classifier." << endl;
        exit(1);
    }

    string fn_haar = string(argv[3]);
    faceClassifier.load(fn_haar);

    fn_haar = string(argv[4]);
    rightEyeClassifier.load(fn_haar);

    fn_haar = string(argv[5]);
    noseClassifier.load(fn_haar);

    read_files(argv[1], "palsy");
    read_files(argv[2], "regular");
    // Get the path to your CSV:
    /*string fn_haar = string(argv[1]);
    string fn_csv = string(argv[2]);
    int deviceId = atoi(argv[3]);
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    cout << "Beginning training. " << endl;
    //model->train(images, labels);
    cout << "Done training. " << endl;
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
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
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
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
            //int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            // Create the text we will annotate the box with:
            //string box_text = format("Prediction = %d", prediction);
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            //putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }*/
    return 0;
}
