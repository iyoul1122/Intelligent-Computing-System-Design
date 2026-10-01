import struct, os
import numpy as np

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
    def __init__():
        pass

    def build_model():
        pass

    def init_model():
        pass

# 网络训练模块
    def forward():
        pass

    def backward():
        pass

    def update():
        pass

    def save_model():
        pass

    def train():
        pass

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