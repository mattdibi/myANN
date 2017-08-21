#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "myANN.cpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    cv::Mat t0 = (cv::Mat_<float>(2,3) << -30,  20, 20, 10, -20, -20);
    cv::Mat t1 = (cv::Mat_<float>(1,3) << -10, 20 , 20);

    vector<cv::Mat> myWeights;
    myWeights.push_back(t0);
    myWeights.push_back(t1);

    vector<int> myLayers;
    myLayers.push_back(2);
    myLayers.push_back(2);
    myLayers.push_back(1);

    myANN ann(myWeights, myLayers);

    cv::Mat d0 = (cv::Mat_<float>(2,1) <<  0 , 0);
    cv::Mat d1 = (cv::Mat_<float>(2,1) <<  0 , 1);
    cv::Mat d2 = (cv::Mat_<float>(2,1) <<  1 , 0);
    cv::Mat d3 = (cv::Mat_<float>(2,1) <<  1 , 1);

    vector<cv::Mat> testingData;
    testingData.push_back(d0);
    testingData.push_back(d1);
    testingData.push_back(d2);
    testingData.push_back(d3);

    cv::Mat y0 = (cv::Mat_<float>(1,1) << 1);
    cv::Mat y1 = (cv::Mat_<float>(1,1) << 0);
    cv::Mat y2 = (cv::Mat_<float>(1,1) << 0);
    cv::Mat y3 = (cv::Mat_<float>(1,1) << 1);

    vector<cv::Mat> expected;
    expected.push_back(y0);
    expected.push_back(y1);
    expected.push_back(y2);
    expected.push_back(y3);

    vector<cv::Mat> output;

    // Testing predict function
    output = ann.predict(testingData);

    for(int i = 0; i < output.size(); i++) {
        cout << i << ": " << output[i] << "\n";
    }

    // Testing costFunction
    cout << "Cost funct known out: " << ann.costFunction(testingData, expected) << endl;

    // Testing create
    myANN untrainedANN;

    vector<int> ld;
    ld.push_back(2);
    ld.push_back(2);
    ld.push_back(1);

    untrainedANN.create(ld);

    untrainedANN.setActivationFunction(ACTFUNCT_TANH);
    untrainedANN.setLearningRate(0.1);
    untrainedANN.setMaxIterations(40000);

    output = untrainedANN.predict(testingData);

    for(int i = 0; i < output.size(); i++) {
        cout << i << ": " << output[i] << "\n";
    }

    cout << "Cost funct untrained out: " << untrainedANN.costFunction(testingData, expected, COSTFUNCT_LOGM) << endl;

    // Testing training function
    untrainedANN.train(testingData, expected);

    output = untrainedANN.predict(testingData);

    for(int i = 0; i < output.size(); i++) {
        cout << i << ": " << output[i] << "\n";
    }

    cout << "Cost funct trained out: " << untrainedANN.costFunction(testingData, expected, COSTFUNCT_LOGM) << endl;

    untrainedANN.printSettings();

    return 0;
}
