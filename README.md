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

Specs:
- CPU: Intel Core i7 4790K (4 cores/8 threads @ 4.4GHz)
- RAM: 16GB 1600MHz

```sh
Correction rate: 90.68%
Cost function output: 0.643168

Training set size: 25000
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

real	9m42.199s
user	70m53.180s
sys	0m3.432s
```
