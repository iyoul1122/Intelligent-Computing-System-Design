# 基本单元模块
import numpy as np

def show_matrix(mat, name):
    print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))

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