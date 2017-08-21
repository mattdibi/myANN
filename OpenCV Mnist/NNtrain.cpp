// readMNIST.cc
// read MNIST data into double vector, OpenCV Mat, or Armadillo mat
// free to use this code for any purpose
// author : Eric Yuan 
// my blog: http://eric-yuan.me/
// part of this code is stolen from http://compvisionlab.wordpress.com/

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml.hpp"
#include <math.h>
#include <fstream>
#include <iostream>

#define NTRAINING_SAMPLES 60000
#define NTESTING_SAMPLES  10000
#define NLABELS 10 // 0-9
#define NFEATURES 784 // = 28 * 28 pixels

#define TRAINING_SAMPLE_RESIZE 3500

// OPTIONAL TESTING
#define TESTING_SAMPLE_RESIZE 200

//Debug:
#define INDEX 50

using namespace cv;
using namespace cv::ml;
using namespace std;

string integerToString(int n)
{
    string convertedN;
    ostringstream convert;
    convert << n;
    convertedN = convert.str();

    return convertedN;
}

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void readMnist(string filename, vector<cv::Mat> &vec){
    ifstream file (filename.c_str(), ios::binary);
    if (file.is_open())
    {
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

        for(int i = 0; i < number_of_images; ++i)
        {
            cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;

                    file.read((char*) &temp, sizeof(temp));
                    tp.at<uchar>(r, c) = (int) temp;
                }
            }
            vec.push_back(tp);
        }
    }
}

void readMnistLabel(string filename, vector<int> &vec)
{
    ifstream file (filename.c_str(), ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;

            file.read((char*) &temp, sizeof(temp));
            vec[i]= (int)temp;
        }
    }
}

Mat rollVectortoMat(const vector<Mat> &data)
{
    Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32FC1);

    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);                                       
        image_row.convertTo(row_i,CV_32FC1, 1/255.);
    }

    return dst;
}

Mat getLabels(const vector<int> &data,int classes = NLABELS)
{
    Mat labels(data.size(),classes,CV_32FC1, float(0));

    for(int i = 0; i <data.size() ; i++)
    {
        int cls = data[i];  
        labels.at<float>(i,cls) = 1.0;  
    }

    return labels;
}

vector<int> interpretPrediction(Mat predictionMat)
{
    int i;
    float max;
    int maxindex;
    vector<int> results;

    for(int j = 0; j < predictionMat.rows; j++)
    {
        max = predictionMat.at<float>(j,0);
        maxindex = 0;

        for(i = 0; i < predictionMat.cols; i++)
        {
            if(predictionMat.at<float>(j,i) > max)
            {
                max = predictionMat.at<float>(j,i);
                maxindex = i;
            }
        }

        results.push_back(maxindex);
    }

    return results;
}

float correctionRate(vector<int> predictions, vector<int> labels)
{
    float correctPredictions = 0;

    for(int i = 0; i < predictions.size(); i++)
    {
        if(predictions[i] == labels[i])
            correctPredictions++;
    }

    return (correctPredictions*100)/predictions.size();

}

int main()
{
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

    Mat trainData = rollVectortoMat(vecImg);                                      
    Mat trainLabels = getLabels(vecLbl);

    // 4: CREATING ANN
    Mat layers = Mat(3,1,CV_32SC1);

    layers.row(0) = Scalar(NFEATURES);
    layers.row(1) = Scalar(1000);
    layers.row(2) = Scalar(NLABELS);

    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();

    mlp->setLayerSizes(layers);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
    mlp->setBackpropMomentumScale(0.1);
    mlp->setBackpropWeightScale(0.1);
    mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, (int)100000, 1e-6));

    // 5: TRAINING ANN
    cout << "Beginning training.\n";
    mlp->train(trainData, ROW_SAMPLE, trainLabels);
    cout << "Training ended.\n";

    // SAVING NETWORK
    mlp->save("ann" + integerToString(trainData.rows) + "samples.xml");

    /* ****************************************************************************************** */
    /*
    // OPTIONAL: TESTING ANN WITH TESTING SET
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

    Mat testData  = rollVectortoMat(vecImgTst);
    Mat testLabels  = getLabels(vecLblTst);

    // PREDICTION
    Mat output(1, NLABELS, CV_32F);

    cout << "Beginning testing.\n";
    mlp->predict(testData, output);
    cout << "Testing ended.\n";

    // INTERPRETING OUTPUT
    vector<int> predictedDigits = interpretPrediction(output);

    // CHECKING HOW MANY PREDICTIONS ARE CORRECT
    cout << "Correction rate: " << correctionRate(predictedDigits, vecLblTst) << "%\n";
    */
    /* ****************************************************************************************** */

    return 0;
}
