#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "../myANN/myANN.cpp"

using namespace cv;
using namespace std;

#define NTRAINING_SAMPLES 60000
#define NTESTING_SAMPLES  10000
#define NLABELS 10 // 0-9
#define NFEATURES 784 // = 28 * 28 pixels

#define TRAINING_SAMPLE_RESIZE 18000

// OPTIONAL TESTING
#define TESTING_SAMPLE_RESIZE 10000

//Debug:
#define INDEX 50

string integerToString(int n) {
    string convertedN;
    ostringstream convert;
    convert << n;
    convertedN = convert.str();

    return convertedN;
}

int ReverseInt (int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void readMnist(string filename, vector<cv::Mat> &vec){

    ifstream file(filename.c_str(), ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        for(int i = 0; i < number_of_images; ++i) {

            cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);

            for(int r = 0; r < n_rows; ++r) {

                for(int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;

                    file.read((char*) &temp, sizeof(temp));
                    tp.at<uchar>(r, c) = (int) temp;
                }
            }
            vec.push_back(tp);
        }
    }
}

void readMnistLabel(string filename, vector<int> &vec) {

    ifstream file (filename.c_str(), ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        for(int i = 0; i < number_of_images; ++i) {
            unsigned char temp = 0;

            file.read((char*) &temp, sizeof(temp));
            vec[i]= (int)temp;
        }
    }
}

vector<cv::Mat> rollVectortoMat(const vector<Mat> &data) {
    //Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32FC1);
    vector<cv::Mat> dst;

    for(unsigned int i = 0; i < data.size(); i++) {

        Mat image_row = data[i].clone().reshape(1,1);
        Mat image_col;

        cv::transpose(image_row, image_col);

        // Mat row_i = dst.row(i);                                       
        // image_row.convertTo(row_i,CV_32FC1, 1/255.);
        Mat tmp;
        image_col.convertTo(tmp,CV_32FC1, 1/255.);
        dst.push_back(tmp);
    }

    return dst;
}

vector<cv::Mat> getLabels(const vector<int> &data,int classes = NLABELS) {
    //Mat labels(data.size(),classes,CV_32FC1, float(0));
    vector<cv::Mat> labels;

    for(int i = 0; i <data.size() ; i++) {
        // int cls = data[i];  
        // labels.at<float>(i,cls) = 1.0;  
        Mat label(classes,1,CV_32FC1,float(0));
        label.at<float>(data[i]-1,1) = 1.0;

        labels.push_back(label);
    }

    return labels;
}

vector<int> interpretPrediction(vector<cv::Mat> predictionMat) {

    int i;
    float max;
    int maxindex;
    vector<int> results;

    for(int j = 0; j < predictionMat.size(); j++) {
        max = predictionMat[j].at<float>(0,0);
        maxindex = 0;

        for(i = 0; i < predictionMat[j].rows; i++) {
            if(predictionMat[j].at<float>(i,0) > max) {
                max = predictionMat[j].at<float>(i,0);
                maxindex = i;
            }
        }

        results.push_back(maxindex);
    }

    return results;
}

float correctionRate(vector<int> predictions, vector<int> labels) {
    float correctPredictions = 0;

    for(int i = 0; i < predictions.size(); i++) {
        if(predictions[i] == labels[i])
            correctPredictions++;
    }

    return (correctPredictions*100)/predictions.size();

}

int main() {
    // 1: READING AND SAVING TRAINING SET
    string filenameImg = "train-images-idx3-ubyte";

    //read MNIST iamge into OpenCV Mat vector
    vector<Mat> vecImg;
    readMnist(filenameImg, vecImg);

    // Debug:
    // cout << vecImg.size() << endl;
    // imshow("1st", vecImg[INDEX]);
    // waitKey();

    string filenameLbl = "train-labels-idx1-ubyte";

    //read MNIST label into int vector
    vector<int> vecLbl(NTRAINING_SAMPLES);
    readMnistLabel(filenameLbl, vecLbl);

    // Debug:
    // cout << vecLbl.size() << endl;
    // cout << "Img label: " << vecLbl[INDEX] << "\n";

    // 3: CONVERTING TRAINING SET
    // Resizing vector to reduce execution time
    vecImg.resize(TRAINING_SAMPLE_RESIZE);
    vecLbl.resize(TRAINING_SAMPLE_RESIZE);

    vector<cv::Mat> trainData = rollVectortoMat(vecImg);                                      
    vector<cv::Mat> trainLabels = getLabels(vecLbl);

    // 4: CREATING ANN
    vector<int> ld;
    ld.push_back(NFEATURES);
    ld.push_back(200);
    ld.push_back(NLABELS);

    myANN mlp;
    mlp.create(ld);
    mlp.setActivationFunction(ACTFUNCT_SIGM);
    mlp.setLearningRate(0.1);
    mlp.setMaxIterations(350);

    // 5: TRAINING ANN
    cout << "Beginning training.\n";
    mlp.train(trainData, trainLabels);
    cout << "Training ended.\n";

    // TESTING ANN WITH TESTING SET
    // READING AND SAVING TEST SET
    string filenameImgTst = "t10k-images-idx3-ubyte";

    vector<Mat> vecImgTst;
    readMnist(filenameImgTst, vecImgTst);

    string filenameLblTst = "t10k-labels-idx1-ubyte";

    vector<int> vecLblTst(NTESTING_SAMPLES);
    readMnistLabel(filenameLblTst, vecLblTst);

    // CONVERTING TESTING SET
    vecImgTst.resize(TESTING_SAMPLE_RESIZE);
    vecLblTst.resize(TESTING_SAMPLE_RESIZE);

    vector<cv::Mat> testData  = rollVectortoMat(vecImgTst);
    vector<cv::Mat> testLabels  = getLabels(vecLblTst);

    // PREDICTION
    vector<cv::Mat> output;

    cout << "Beginning testing.\n";
    output = mlp.predict(testData);
    cout << "Testing ended.\n";

    // INTERPRETING OUTPUT
    vector<int> predictedDigits = interpretPrediction(output);

    // cout << "Pred out\n" << output[0] << endl;
    // cout << "Pred digit" << predictedDigits[0] << endl;
    // cout << "test label" << vecLblTst[0] << endl;

    // CHECKING HOW MANY PREDICTIONS ARE CORRECT
    cout << "Correction rate: " << correctionRate(predictedDigits, vecLblTst) << "%\n";
    cout << "Cost function output: " << mlp.costFunction(testData, testLabels, COSTFUNCT_LOGM) << endl;;

    cout << "\nTraining set size: " << TRAINING_SAMPLE_RESIZE << endl;
    mlp.printSettings();

    return 0;
}
