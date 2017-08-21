#include "myANN.h"

myANN::myANN() {
    trained = false;
    created = false;
}

myANN::myANN(vector<cv::Mat> wms, vector<int> ls) {
    weightMatrix = wms;
    layersDescription = ls;

    trained = false;
    created = false;
}

/**
    It calculates the activation function element-wise on a vector passed as argument

    @param v a non empty opencv Mat vector
    @return an opencv Mat vector

*/
cv::Mat myANN::computeActivationFunction(cv::Mat v) {
    cv::Mat res = cv::Mat(v.rows, 1, CV_32FC1);

    if(v.cols != 1) {
        cout << "Assertion v.cols == 1 failed!\n";
        cout << "ERROR: Sigmf received a matrix instead of a vector!\n";
    } else {
        if(activationFunction == ACTFUNCT_TANH) {
            for(int i = 0; i < v.rows; i++) {
                res.at<float>(i,0,0) = tanh(v.at<float>(i,0,0));
            }
        } else {
            for(int i = 0; i < v.rows; i++) {
                res.at<float>(i,0,0) = 1 / (1 + exp(-1 * v.at<float>(i,0,0)));
            }
        }
    }

    return res;
}

/**
    It copy a vector and add the bias unit in the first location

    @param src the source vector
    @return an opencv mat vector with a bias unit
*/
cv::Mat myANN::copyAddBias(cv::Mat src) {
    cv::Mat dst = cv::Mat(src.rows + 1, 1, CV_32FC1);

    dst.at<float>(0,0,0) = 1;

    for(int i = 0; i < src.rows; i++) {
        dst.at<float>(i+1,0,0) = src.at<float>(i,0,0);
    }

    return dst;
}

/**
    It copy a vector and delete the bias unit in the first location

    @param src the source vector
    @return an opencv mat vector without the bias unit
*/
cv::Mat myANN::copyDelBias(cv::Mat src) {
    cv::Mat dst = cv::Mat(src.rows - 1, 1, CV_32FC1);

    for(int i = 0; i < dst.rows; i++) {
        dst.at<float>(i,0,0) = src.at<float>(i+1,0,0);
    }

    return dst;
}

/**
    It calculate the outout of the neural network using the Mat as the input
    and the Forward propagation algorithm

    @param data a Mat vector containing the values of all layers in the Network
    @return a vector of Mat elements as the outputs
*/
vector<cv::Mat> myANN::forwardPropagate(cv::Mat input) {
    vector<cv::Mat> layer;

    if(input.empty()) {
        cout << "Assertion input.size() != 0 failed!\n";
        cout << "ERROR: Empty input vector!\n";
    } else if(input.cols != 1) {
        cout << "Assertion input[0].cols == 1 failed!\n";
        cout << "ERROR: input must be a vector!\n";
    } else if(input.rows != layersDescription[0]) {
        cout << "Assertion data[0].rows == layersDescription[0] - 1 failed!\n";
        cout << "ERROR: data.rows and input layer units must agree!\n";
    } else {
        // Creating layer vector
        for(int i = 0; i < layersDescription.size(); i++) {
            int rows;

            if(i < layersDescription.size() - 1)
                rows = layersDescription[i] + 1;
            else
                rows = layersDescription[i];

            cv::Mat tmp = cv::Mat::zeros(rows, 1, CV_32FC1);

            layer.push_back(tmp);
        }

        //Loading data: layers[0] = input + bias unit;
        layer[0] = copyAddBias(input);

        for(int i = 0; i < layer.size() - 1; i++) {
            // TODO: Insert here hardware accelerated function for matrix multipl
            // z(l+1) = W(l)*a(l);
            cv::Mat z = weightMatrix[i] * layer[i];

            // Activation function
            // a(l) = g(z(l));
            cv::Mat a = computeActivationFunction(z);

            // Copying vector adding bias unit if not the output layer
            if( i < layersDescription.size() - 2)
                layer[i+1] = copyAddBias(a);
            else
                layer[i+1] = a;
        }

    }

    return layer;
}

/**
    It creates the matrices and vector that implements an Artificial Neural Network

    @param layersDescriptors a vector of int with the layer description (layer[0] = # of input units)
    @param activationFunct the desired activation function to be used
    @param initializationBoundaries range of numbers to be randomly generated to initialize matrices
    @return void

*/
void myANN::create(vector<int> layersDescr, int initializationBoundaries = DEFAULT_BOUNDARIES) {

    if(layersDescr.empty()) {
        cout << "Assertion layersDescr.size() != 0 failed!\n";
        cout << "ERROR: Empty layers descriptors vector!\n";
        return;
    }

    // Making sure that we have empty vectors
    weightMatrix.clear();

    // Filling weight matrices
    for(int k = 0; k < layersDescr.size() - 1; k++) {
        // Random initialization
        cv::Mat tmp = cv::Mat(layersDescr[k+1], layersDescr[k] + 1, CV_32FC1);
        cv::randu(tmp, cv::Scalar::all(-1*initializationBoundaries), cv::Scalar::all(initializationBoundaries));

        weightMatrix.push_back(tmp);
    }

    // Filling costFunctionDerivative
    for(int l = 0; l < layersDescr.size() - 1; l++) {
        cv::Mat tmp = cv::Mat::zeros(layersDescr[l+1], layersDescr[l] + 1, CV_32FC1);

        costFunctionDerivative.push_back(tmp);
    }

    trained = false;
    created = true;

    maxIterations = DEFAULT_ITERATIONS;
    learningRate = DEFAULT_LEARNINGRATE;
    activationFunction = ACTFUNCT_SIGM;

    layersDescription = layersDescr;

    return;
}

/**
    It calculate the correct weightMatrices using the backpropagation algorith

    @param data a vector of Mat elements which contains the input data
    @param expectedOutput a vector of Mat elements which contains the labels for each input data
    @param maxIterations maximum number of iteration on the same data set
    @param learningRate the learning rate of the backpropagation algorithm
    @return void
*/
void myANN::train(vector<cv::Mat> data, vector<cv::Mat> expectedOutput) {

    if(data.empty()) {
        cout << "Assertion data.size() != 0 failed!\n";
        cout << "ERROR: Empty data vector!\n";
    } else if(data[0].cols != 1) {
        cout << "Assertion data[0].cols == 1 failed!\n";
        cout << "ERROR: data must be a vector!\n";
    } else if(data[0].rows != layersDescription[0]) {
        cout << "Assertion data[0].rows == layers[0].rows failed!\n";
        cout << "ERROR: data.rows and input layer units must agree!\n";
    } else if(expectedOutput[0].cols != 1) {
        cout << "Assertion expectedOutput[0].cols == 1 failed!\n";
        cout << "ERROR: expectedOutput must be a vector!\n";
    } else if(expectedOutput[0].rows != layersDescription[layersDescription.size()-1]) {
        cout << "Assertion expectedOutput[0].rows == layers[output_layer].rows failed!\n";
        cout << "ERROR: expectedOutput.rows and output layer units dimensions must agree!\n";
    } else if(!created) {
        cout << "Assertion isCreated() failed!\n";
        cout << "ERROR: Neural Network must be initialized with create() function!\n";
    } else {
        for(int iterations = 0; iterations < maxIterations; iterations++) {
            int i;

            #pragma omp parallel for private(i)
            for(i = 0; i < data.size(); i++) {   
                // // ForwardPropagation
                vector<cv::Mat> output = forwardPropagate(data[i]);

                // BackwardPropagation
                // TODO: transform into a separate function (?)
                /* ******************************************************************* */
                // Initialize errorLayers vector
                vector<cv::Mat> errorLayers;

                for(int j = 0; j < layersDescription.size(); j++) {
                    int rows;

                    if(j < layersDescription.size() - 1)
                        rows = layersDescription[j] + 1;
                    else
                        rows = layersDescription[j];

                    cv::Mat tmp = cv::Mat::zeros(rows, 1, CV_32FC1);

                    errorLayers.push_back(tmp);
                }

                // Initializing output layer
                errorLayers[layersDescription.size()-1] = output[layersDescription.size()-1] - expectedOutput[i];

                // Error back propagation
                for(int l = layersDescription.size() - 1; l > 1; l--) {
                    cv::Mat transposedWeightMatrix;
                    cv::transpose(weightMatrix[l - 1], transposedWeightMatrix);

                    cv::Mat activationFunctionDerivative;
                    
                    if(activationFunction == ACTFUNCT_SIGM) {
                        // activationFunctionDerivative = layers[l-1].*(1-layers[l-1]); FOR ACTFUNCT_SIGM
                        cv::multiply(output[l-1], (1-output[l-1]), activationFunctionDerivative);
                    } else if(activationFunction == ACTFUNCT_TANH) {
                        // activationFunctionDerivative = (1 - tanh(layers[l-1])^2); FOR ACTFUNCT_TANH
                        cv::multiply(computeActivationFunction(output[l-1]),computeActivationFunction(output[l-1]),activationFunctionDerivative);
                        activationFunctionDerivative = 1 - activationFunctionDerivative;
                        // TODO: this is not so beautiful
                    }

                    // Output layer doesn't need discarding bias unit
                    if(l == layersDescription.size() - 1)
                        cv::multiply(transposedWeightMatrix*errorLayers[l],activationFunctionDerivative,errorLayers[l-1]);
                    else
                    {
                        // Discarding bias unit delta: d = errorLayers[l] - biasUnit;
                        cv::Mat d = copyDelBias(errorLayers[l]);
                        cv::multiply(transposedWeightMatrix*d,activationFunctionDerivative,errorLayers[l-1]);
                    }
                    
                }

                for(int k = 0; k < layersDescription.size() - 1; k++) {
                    cv::Mat transposedLayer;
                    cv::transpose(output[k], transposedLayer);

                    // Output layer doesn't need discarding bias unit
                    if( k < layersDescription.size() - 2) {
                        // Discarding bias unit delta: d = errorLayers[k+1] - biasUnit;
                        cv::Mat d = copyDelBias(errorLayers[k+1]);
                        costFunctionDerivative[k] += d * transposedLayer;
                    }
                    else
                        costFunctionDerivative[k] += errorLayers[k+1] * transposedLayer;
                }
                /* ******************************************************************* */
            }

            for(int l = 0; l < layersDescription.size() - 1; l++) {
                // Divide costFunctionDerivative for the number of examples
                costFunctionDerivative[l] = costFunctionDerivative[l]/data.size();

                // Update network weights
                weightMatrix[l] = weightMatrix[l] - learningRate*costFunctionDerivative[l];
            }

            cout << " Progress: " << iterations << "iter/" << maxIterations << "\r" << flush;

        }

        // Add endline to not overwrite progress indicator
        cout << endl;

        trained = true;

    }

    return;
}

/**
    It calculate the outout of the neural network using the vector of Mat as the input
    and the Forward propagation algorithm

    @param data a vector of Mat elements
    @return a vector of Mat elements as the outputs
*/
vector<cv::Mat> myANN::predict(vector<cv::Mat> data) {
    vector<cv::Mat> output(data.size());

    if(data.empty()) {
        cout << "Assertion data.size() != 0 failed!\n";
        cout << "ERROR: Empty data vector!\n";
    } else if(data[0].cols != 1) {
        cout << "Assertion data[0].cols == 1 failed!\n";
        cout << "ERROR: data must be a vector!\n";
    } else if(data[0].rows != layersDescription[0]) {
        cout << "Assertion data[0].rows == layers[0].rows failed!\n";
        cout << "ERROR: data.rows and input layer units must agree!\n";
    } else {
        // For each input compute output layer
        #pragma omp parallel for
        for(int j = 0; j < data.size(); j++) {
            // Forward propagate input
            vector<cv::Mat> layersOutput = forwardPropagate(data[j]);

            // Saving output layer
            output[j] = layersOutput[layersDescription.size()-1];
        }

    }

    return output;
}

/**
    It calculate the outout of the cost function for the artificial neural network
    Note: it uses the predict() function

    @param data a vector of Mat elements which contains the input data
    @param expectedOutput a vector of Mat elements which contains the labels for each input data
    @return the value of the cost function
*/
float myANN::costFunction(vector<cv::Mat> data, vector<cv::Mat> expectedOutput, bool costFunctionType = COSTFUNCT_QUAD) {
    float result = 0;

    if(data.empty()) {
        cout << "Assertion data.size() != 0 failed!\n";
        cout << "ERROR: Empty data vector!\n";
    } else if(data[0].cols != 1) {
        cout << "Assertion data[0].cols == 1 failed!\n";
        cout << "ERROR: data must be a vector!\n";
    } else if(data[0].rows != layersDescription[0]) {
        cout << "Assertion data[0].rows == layers[0].rows failed!\n";
        cout << "ERROR: data.rows and input layer units must agree!\n";
    } else if(expectedOutput[0].cols != 1) {
        cout << "Assertion expectedOutput[0].cols == 1 failed!\n";
        cout << "ERROR: expectedOutput must be a vector!\n";
    } else if(expectedOutput[0].rows != layersDescription[layersDescription.size()-1]) {
        cout << "Assertion expectedOutput[0].rows == layers[output_layer].rows failed!\n";
        cout << "ERROR: expectedOutput.rows and output layer units dimensions must agree!\n";
    } else {

        vector<cv::Mat> output = predict(data);

        // Quadratic cost function implementation
        if(costFunctionType == COSTFUNCT_QUAD) {
            for(int i = 0; i < data.size(); i++) {
                for(int j = 0; j < layersDescription[layersDescription.size()-1]; j++) {
                    result += pow(output[i].at<float>(j,0,0) - expectedOutput[i].at<float>(j,0,0),2);
                }
            }

            result = result / data.size();
        } else if(costFunctionType == COSTFUNCT_LOGM) { // Log cost function implementation
            for(int i = 0; i < data.size(); i++) {
                for(int j = 0; j < layersDescription[layersDescription.size()-1]; j++) {
                    result += expectedOutput[i].at<float>(j,0,0)*log(output[i].at<float>(j,0,0))
                                + (1 - expectedOutput[i].at<float>(j,0,0))*(log(1 - output[i].at<float>(j,0,0)));
                    
                }
            }

            result = -1 * (result / data.size());
        } else {
            cout << "ERROR: Unkown cost function type!\n";
        }
    }

    return result;
}

/**
    It prints all the artificial neural network settings and matrix dimensions
*/
void myANN::printSettings() {
    cout << "myANN Settings:\n";
    
    if(activationFunction == ACTFUNCT_SIGM)
        cout << "- Activation function: SIGMOIDAL\n";
    else if(activationFunction == ACTFUNCT_TANH)
        cout << "- Activation function: TANH\n";
    else
        cout << "- Activation function: *UNKOWN* an error occurred!\n";

    cout << "- Max Iterations: " << maxIterations << endl;
    cout << "- Learning Rate: " << learningRate << endl;

    cout << "- Number of layers: " << layersDescription.size() << endl;
    
    for(int i = 0; i < layersDescription.size(); i++) {
        cout << "-  Layer " << i << " layer dimension: " << layersDescription[i] << endl;
    }

    cout << "- Number of matrices: " << weightMatrix.size() << endl;

    for(int i = 0; i < weightMatrix.size(); i++) {
        cout << "-  Layer " << i << " matrix dimension: " << weightMatrix[i].cols << "x" << weightMatrix[i].rows << endl;
    }

    return;
}

