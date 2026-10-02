# 基本单元模块
import numpy as np

# 全连接层
class FullyConnectedLayer(object):
    # 全连接层初始化
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
    # 参数初始化
    def init_param(self, std = 0.01):
        self.weight = np.random.normal(loc=0.0, scale=std,
                                      size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    # 前向传播的计算
    def forward(self, input):
        self.input = input  # 缓存输入
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output
    # 反向传播的计算
    def backward(self, top_diff):
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff
    # 参数更新
    def update_param(self, lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias
    # 参数加载
    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias
    # 参数保存
    def save_param(self):
        return self.weight, self.bias

# ReLU激活层
class ReLULayer(object):
    # 前向传播的计算
    def forward(self, input):
        self.input = input  # 缓存输入
        output = np.maximum(0, self.input)
        return output
    # 反向传播的计算
    def backward(self, top_diff):
        bottom_diff = top_diff * (self.input >= 0)
        return bottom_diff

# Softmax损失层
class SoftmaxLossLayer(object):
    # 前向传播的计算
    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob
    # 计算损失
    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    # 反向传播的计算
    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff