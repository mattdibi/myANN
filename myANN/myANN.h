/**
    myANN
    Purpose: Creating an artificial neural network implementation to be easily
    ported in SDSoC Developement Environment and hardware accelerated on Xilinx FPGAs

    @author Mattia Dal Ben
    @version 1.0 26/01/2017

*/

#include <vector>
#include <opencv2/opencv.hpp>

//Debugging
#include <stdio.h>

using namespace std;

#define INITMET_ZEROS 0
#define INITMET_ONES 1
#define INITMET_RAND 2

#define DEFAULT_ITERATIONS 1000
#define DEFAULT_LEARNINGRATE 0.1
#define DEFAULT_BOUNDARIES 1

#define ACTFUNCT_SIGM 0
#define ACTFUNCT_TANH 1

#define COSTFUNCT_QUAD 0
#define COSTFUNCT_LOGM 1

class myANN {

  public:
    // Selectors
    int LayerVectDim() {return layersDescription.size();};
    int MatrixVectDim() {return weightMatrix.size();};
    cv::Mat WeightMatrix(int k) {return weightMatrix[k];};

    char activationFunctionType() {return activationFunction;};

    bool isTrained() {return trained;};
    bool isCreated() {return created;};

    // Constructor
    myANN();
    myANN(vector<cv::Mat> wms, vector<int> ls);

    // Methods
    void create(vector<int> layersDescr, int initializationBoundaries);
    void train(vector<cv::Mat> data, vector<cv::Mat> expectedOutput);
    vector<cv::Mat> predict(vector<cv::Mat> data);

    float costFunction(vector<cv::Mat> data, vector<cv::Mat> expectedOutput, bool costFunctionType);

    // Settings
    void setActivationFunction(char activationFunct){activationFunction = activationFunct;};
    void setMaxIterations(unsigned int maxIter){maxIterations = maxIter;};
    void setLearningRate(float learningR){learningRate = learningR;};

    // Utilites
    void printSettings();

  private:
    // Neural network core
    vector<int> layersDescription;
    vector<cv::Mat> weightMatrix;

    // Backpropagation algorithm
    vector<cv::Mat> costFunctionDerivative;

    // Neural Network parameters
    char activationFunction;
    unsigned int maxIterations;
    float learningRate;

    // Flags
    bool trained;
    bool created;


    //Private functions
    cv::Mat copyAddBias(cv::Mat src);
    cv::Mat copyDelBias(cv::Mat src);

    // Activation functions
    cv::Mat computeActivationFunction(cv::Mat v);
    vector<cv::Mat> forwardPropagate(cv::Mat input);

};
