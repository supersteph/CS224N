import numpy as np
import math

x_conv = np.array([0.1,0.2,0.3])
W_gate = np.array([[-0.1308,  0.3081,  0.5199],
        [-0.3833,  0.3156,  0.3498],
        [ 0.1566, -0.0406, -0.3571]])
b_gate = np.array([-0.0569,  0.1396,  0.3328])
W_proj = np.array([[-0.0824, -0.0009,  0.4296],
        [ 0.0214, -0.3855,  0.3013],
        [ 0.5264, -0.5759,  0.1331]])
b_proj = np.array([0.5649, 0.2499, 0.5122])


# custom function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def relu(x):
  return max(0,x)
# define vectorized sigmoid
sigmoid_v = np.vectorize(sigmoid)
relu_v = np.vectorize(relu)
proj = relu_v(np.matmul(W_proj,x_conv)+b_proj)
gate = sigmoid_v(np.matmul(W_gate,x_conv)+b_gate)
print(np.multiply(proj,gate)+np.multiply(1-gate,x_conv))