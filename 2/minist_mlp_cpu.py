import struct, os
import numpy as np

import layers

MNIST_DIR = ""
TRAIN_DATA = ""
TRAIN_LABEL = ""
TEST_DATA = ""
TEST_LABEL = ""

# 数据加载模块
def load_mnist(self, fire_dir, is_images=True):
    bin_file = open(fire_dir, 'rb') # 'rb' 得到 bytes 对象
    bin_data = bin_file.read()
    bin_file.close()

    # struct.unpack_from(fmt, buffer, offset)：按 fmt 的格式从 buffer 的第 offset 个字节开始解包
    # 图像文件
    if is_images:
        fmt_header = '>iiii'
        magic, num_image, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    # 标签文件
    else:
        fmt_header = '>ii'
        magic, num_image = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1

    data_size = num_image * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B',   # i 表示4字节有符号 int，B 表示 1字节无符号 int。整字段 'B' 前可以加数字表示重复次数。
                                  bin_data, struct.calcsize(fmt_header))
    mat_data = np.reshape(mat_data, [num_image, num_rows * num_cols])
    return mat_data

def load_data(self):
    train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
    train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
    test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
    test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
    self.train_data = np.append(train_images, train_labels, axis=1) # 拼接图像数据和标签
    self.tast_data = np.append(test_images, test_labels, axis=1)

# 网络结构模块
class MNIST_MLP(object):
    # 神经网络初始化
    def __init__(self, batch_size=100, input_size=784, hidden1=32, hidden2=16,
                 out_classes=10, lr=0.01, max_epoch=2, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
    # 建立网络结构
    def build_model(self):
        print("Building multi-layer perception model...")
        self.fc1 = layers.FullyConnectedLayer(num_input=self.input_size, num_output=self.hidden1)
        self.relu1 = layers.ReLULayer()
        self.fc2 = layers.FullyConnectedLayer(num_input=self.hidden1, num_output=self.hidden2)
        self.relu2 = layers.ReLULayer()
        self.fc3 = layers.FullyConnectedLayer(num_input=self.hidden2, num_output=self.out_classes)
        self.softmax = layers.SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]
    # 网络参数初始化
    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param()

# 网络训练模块
    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        prob = self.softmax(h3)
        return prob

    def backward(self):
        dloss = self.softmax.backward()
        dh3 = self.fc3.backward(dloss)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def save_model(self, param_dir):
        params = []
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        np.savez(param_dir, params)

    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size
        for idx_epoch in range(self.max_epoch):
            mlp.shuffle_data()
            for idx_batch in range(self.max_batch):
                batch_images = self.train_data[idx_batch * self.batch_size :(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size :(idx_batch + 1) * self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))

# 网络推理模块
    def load_model():
        pass

    def evaluate():
        pass

# 模型构建函数
def build_mnist_mlp():
    pass

# 主函数
if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()