#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace cv;
using namespace std;

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
            if(neg > 100){
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

Mat normalize_to_1D(String imgname) {
  Mat img_mat = imread(imgname,0); // I used 0 for greyscale
  Mat training_mat(1, img_mat.rows * img_mat.cols,CV_32FC1);
  int file_num = 0;
  int ii = 0; // Current column in training_mat
  for (int i = 0; i<img_mat.rows; i++) {
    for (int j = 0; j < img_mat.cols; j++) {
      training_mat.at<float>(file_num,ii++) = img_mat.at<uchar>(i,j);
    }
  }
  return training_mat;
}

int main(int argc, char** argv)
{
  Mat training_image;
  training_image = normalize_to_1D("/Users/Bunchhieng/Documents/Bunchhieng/HiveLabs/Baymax/Stroke-Detector/Data/filtered-images/train/palsy/000113.jpg");
  int labels[1] = {1};
  Mat labelsMat(1,1,CV_32S, labels);
  // if (argc != 5) {
  //     cout << "usage: " << argv[0] << "</path/to/training/csv.ext> </path/to/test/csv.ext>" << endl;
  //     cout << "\t </path/to/training/csv.ext> -- Path to the CSV file with the training face database." << endl;
  //     cout << "\t </path/to/cv/csv.ext> -- Path to the CSV file with the cross validation face database." << endl;
  //     cout << "\t </path/to/test/csv.ext> -- Path to the CSV file with the test face database." << endl;
  //     exit(1);
  // }
  // // Get the path to your CSV:
  // string fn_training = string(argv[3]);
  // string fn_cv = string(argv[4]);
  // string fn_test = string(argv[5]);
  //
  // // These vectors hold the images and corresponding labels:
  // vector<Mat> training_images;
  // vector<int> training_labels;
  //
  // vector<Mat> cv_images;
  // vector<int> cv_labels;
  //
  // vector<Mat> test_images;
  // vector<int> test_labels;
  //
  // try {
  //     read_csv(fn_training, training_images, training_labels);
  //     read_csv(fn_cv, cv_images, cv_labels);
  //     read_csv(fn_test, test_images, test_labels);
  // } catch (cv::Exception& e) {
  //     cerr << "Error opening csv file . Reason: " << e.msg << endl;
  //     // nothing more we can do
  //     exit(1);
  // }

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


    Ptr<ml::SVM> svm = ml::SVM::create();
  // edit: the params struct got removed,
  // we use setter/getter now:
  svm->setType(ml::SVM::C_SVC);
  svm->setKernel(ml::SVM::LINEAR);
  svm->setGamma(3);
  svm->train( training_image , ml::ROW_SAMPLE , labelsMat );
  float shit = svm->predict(training_image);
  cout << shit << endl;
    // Vec3b green(0,255,0), blue (255,0,0);
    // // Show the decision regions given by the SVM
    // for (int i = 0; i < image.rows; ++i)
    //     for (int j = 0; j < image.cols; ++j)
    //     {
    //         Mat sampleMat = (Mat_<float>(1,2) << j,i);
    //         float response = SVM.predict(sampleMat);
    //
    //         if (response == 1)
    //             image.at<Vec3b>(i,j)  = green;
    //         else if (response == -1)
    //              image.at<Vec3b>(i,j)  = blue;
    //     }

    // // Show the training data
    // int thickness = -1;
    // int lineType = 8;
    // circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
    // circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
    // circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    // circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
    //
    // // Show support vectors
    // thickness = 2;
    // lineType  = 8;
    // int c     = SVM.get_support_vector_count();
    //
    // for (int i = 0; i < c; ++i)
    // {
    //     const float* v = SVM.get_support_vector(i);
    //     circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    // }
    //
    // imwrite("result.png", image);        // save the image
    //
    // imshow("SVM Simple Example", image); // show it to the user
    // waitKey(0);
    return 0;
}
