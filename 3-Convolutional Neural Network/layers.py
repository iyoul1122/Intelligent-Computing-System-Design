# 基本单元模块

import numpy as np

def show_matrix(mat, name):
    print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))

# 卷积层
class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        print("\tConvolutional layer with kernel size %d, input channel %d, output channel %d, padding %d, stride %d." % (
            self.kernel_size, self.channel_in, self.channel_out, self.padding, self.stride))

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std,
                                       size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])

    def forward(self, input):
        self.input = input
        height = self.input.shape[2] + 2*self.padding
        width = self.input.shape[3] + 2*self.padding
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :,
                       self.padding:self.padding+self.input.shape[2],
                       self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size)//self.stride + 1
        width_out = (width - self.kernel_size)//self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        self.output[idxn, idxc, idxh, idxw] = np.sum(
                            self.input_pad[idxn, :,
                                           idxh*self.stride:idxh*self.stride+self.kernel_size,
                                           idxw*self.stride:idxw*self.stride+self.kernel_size]
                            * self.weight[:, :, :, idxc]) + self.bias[idxc]
        return self.output

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

# 最大池化层
class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        print('\tMax pooling layer with kernel size %d, stride %d.' % (
            self.kernel_size, self.stride))

    def forward(self, input):
        self.input = input  #[N,C,H,W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size)//self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size)//self.stride + 1
        self.output = np.zeros([self.input.shape[0] , self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        self.output[idxn, idxc, idxh, idxw] = np.max(
                            self.input[idxn, idxc,
                                       idxh*self.stride:idxh*self.stride+self.kernel_size,
                                       idxw*self.stride:idxw*self.stride+self.kernel_size])
        return self.output

# flatten层
class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (
            str(self.input_shape), str(self.output_shape)))

    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        self.input = input.transpose(0,2,3,1)
        self.output = self.input.reshape([input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out')
        return self.output

# 全连接层
class FullyConnectedLayer(object):
    # 全连接层初始化
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    # 参数初始化
    def init_param(self, std = None):
        # He 初始化：std = sqrt(2 / fan_in)，按本层输入维度自适应；显式传入 std 时沿用传入值
        if std is None:
            std = np.sqrt(2.0 / self.num_input)
        self.weight = np.random.normal(loc=0.0, scale=std,
                                      size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')
        # Adam 状态
        self.m_w = np.zeros_like(self.weight)
        self.v_w = np.zeros_like(self.weight)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        self.t = 0            # 该层被更新的次数
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
    def update_param(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        # 权重
        self.m_w = beta1 * self.m_w + (1 - beta1) * self.d_weight
        self.v_w = beta2 * self.v_w + (1 - beta2) * (self.d_weight ** 2)
        m_w_hat = self.m_w / (1 - beta1 ** self.t)
        v_w_hat = self.v_w / (1 - beta2 ** self.t)
        self.weight -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

        # 偏置
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.d_bias
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.d_bias ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)
        self.bias -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
    # 参数加载
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')
    # 参数保存
    def save_param(self):
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')
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
        loss = -np.sum(np.log(self.prob + 1e-12) * self.label_onehot) / self.batch_size
        return loss
    # 反向传播的计算
    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff