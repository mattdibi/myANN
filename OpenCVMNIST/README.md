### Results

Specs:
- CPU: Intel Core i7 4790K (4 cores/8 threads @ 4.4GHz)
- RAM: 16GB 1600MHz

#### Test 1
```sh
Correction rate: 90.76%

Training set size: 18000
myANN Settings:
- Activation function: SIGMOIDAL
- Max Iterations: 350
- Learning Rate: 0.1
- Number of layers: 3
-  Layer 0 layer dimension: 784
-  Layer 1 layer dimension: 200
-  Layer 2 layer dimension: 10

real	23m30.504s
user	23m29.828s
sys	0m0.572s
```

#### Test 2
```sh
Correction rate: 72.99%

Training set size: 25000
myANN Settings:
- Activation function: SIGMOIDAL
- Max Iterations: 350
- Learning Rate: 0.75
- Number of layers: 3
-  Layer 0 layer dimension: 784
-  Layer 1 layer dimension: 200
-  Layer 2 layer dimension: 10

real	35m49.767s
user	35m48.648s
sys	0m0.832s
```
