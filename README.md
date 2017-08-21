# myANN
My Artificial Neural Network.

I used this project to learn OpenCV basics and to have a better understanding of how an Artificial Neural Network works.
The network characteristics:
- Forward propagation
- Optimization: backpropagation algorithm w/ gradient descent

### Requirements
- cmake
- OpenCV

### Results
```sh
Correction rate: 90.61%
Cost function output: 0.652445

Training set size: 18000
myANN Settings:
- Activation function: SIGMOIDAL
- Max Iterations: 350
- Learning Rate: 0.75
- Number of layers: 3
-  Layer 0 layer dimension: 784
-  Layer 1 layer dimension: 200
-  Layer 2 layer dimension: 10
- Number of matrices: 2
-  Layer 0 matrix dimension: 785x200
-  Layer 1 matrix dimension: 201x10
```
