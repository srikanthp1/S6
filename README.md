# Backpropagation 

## part 1


### explanation 

![alt text](https://github.com/srikanthp1/S6/blob/main/images/1.png?raw=true)

----------
step by step 
----------

h1 = w1*i1 + w2*i2
h2 = w3*i1 + w4*i2
a_h1 = σ(h1) = 1/(1 + exp(-h1))
a_h2 = σ(h2)
o1 = w5*a_h1 + w6*a_h2
o2 = w7*a_h1 + w8*a_h2
a_o1 = σ(o1)
a_o2 = σ(o2)
E_total = E1 + E2
E1 = ½ * (t1 - a_o1)²
E2 = ½ * (t2 - a_o2)²

∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1
∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2

∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1
∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2
∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1
∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2

* we follow chain rule to find partial derivatives 
* loss is half of square of difference. 
* total loss would be sum of loss of each final output node.
* the deeper the network is the longer equation is going to be. 
* if not careful may lead to vanishing gradients(sigmoid-(0,1)range)(multiplying multiple numbers less than 1)

### lr experiments 

- lr = 0.1
![alt text](https://github.com/srikanthp1/S6/blob/main/images/lr01.png?raw=true)

- lr = 0.2
![alt text](https://github.com/srikanthp1/S6/blob/main/images/lr02.png?raw=true)

- lr = 0.5
![alt text](https://github.com/srikanthp1/S6/blob/main/images/lr05.png?raw=true)

- lr = 0.8
![alt text](https://github.com/srikanthp1/S6/blob/main/images/lr08.png?raw=true)

- lr = 1
![alt text](https://github.com/srikanthp1/S6/blob/main/images/lr10.png?raw=true)

- lr = 2
![alt text](https://github.com/srikanthp1/S6/blob/main/images/lr20.png?raw=true)

* it takes longer to converge if lr is small 

## part 2

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
           Dropout-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             584
              ReLU-6            [-1, 8, 28, 28]               0
       BatchNorm2d-7            [-1, 8, 28, 28]              16
            Conv2d-8           [-1, 20, 28, 28]           1,460
              ReLU-9           [-1, 20, 28, 28]               0
      BatchNorm2d-10           [-1, 20, 28, 28]              40
        MaxPool2d-11           [-1, 20, 14, 14]               0
           Conv2d-12           [-1, 16, 14, 14]             336
             ReLU-13           [-1, 16, 14, 14]               0
      BatchNorm2d-14           [-1, 16, 14, 14]              32
          Dropout-15           [-1, 16, 14, 14]               0
           Conv2d-16           [-1, 16, 14, 14]           2,320
             ReLU-17           [-1, 16, 14, 14]               0
      BatchNorm2d-18           [-1, 16, 14, 14]              32
           Conv2d-19           [-1, 30, 14, 14]           4,350
             ReLU-20           [-1, 30, 14, 14]               0

```

* I used nn.sequensial module to define each block of convulution 
* followed the architecture discussed but with limited channels 
* made sure RF is more than 32*32 
* used 7*7 RF for 1st block 
* used 1x1 aggregators and less channels for improving accuracy with less number of channels 


[**NetworkRF**](https://github.com/srikanthp1/S6/blob/main/S6b.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/srikanthp1/S6/blob/main/S6b.ipynb)