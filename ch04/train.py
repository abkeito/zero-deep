import numpy as np
import matplotlib.pyplot as plt

def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのパスを追加
from dataset_zero.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

## 4.3 数値微分
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

dy = numerical_diff(function_1, 5)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.plot(5, function_1(5), "ro")
plt.plot(x, dy*(x-5)+function_1(5), linestyle="--")
plt.title("numerical diff")
plt.savefig("function_1.png")

# 4.4 勾配法
def _numeriacal_gradient_no_batch(f, x):
    h = 1e-4
    grd = np.zeros_like(x) # xと同じ形状の配列を生成
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x + h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x - h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grd[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grd

def numeriacal_gradient(f, x):
    if x.ndim == 1:
        return _numeriacal_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(x)
        for i in range(x.shape[0]):
            grad[i] = _numeriacal_gradient_no_batch(f, x[i])
        return grad

def function_2(x):
    return np.sum(x ** 2)

# 勾配の図示
x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

grad = numeriacal_gradient(function_2, np.array([X, Y]).T).T

plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1], angles='xy', color='r')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("x0")
plt.ylabel("x1")
plt.grid()
plt.draw()
plt.savefig("gradient.png")

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numeriacal_gradient(f, x)
        x -= lr * grad
    return x

x = np.array([-3.0, 4.0])
print("Initial x:", x)
x = gradient_descent(function_2, x, lr=0.1, step_num=100)
print("x:", x)

# 4.5 ミニバッチ学習
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def soft_max(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = soft_max(a2)

        return y
    
    # x: 訓練データ、t: 教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x ,t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numeriacal_gradient(loss_W, self.params['W1'])
        grads['b1'] = numeriacal_gradient(loss_W, self.params['b1'])
        grads['W2'] = numeriacal_gradient(loss_W, self.params['W2'])
        grads['b2'] = numeriacal_gradient(loss_W, self.params['b2'])
        return grads

# 実際に学習する
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
# hyper parameters
iters_num = 100
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# 1エポックあたりの繰り返し
iter_per_epoch = max(10, 1)
for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配を求める
    grad = network.numerical_gradient(x_batch, t_batch)

    # パラメータ更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    print(f"Loss at {i}: {loss}")
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Train acc: {train_acc}, Test acc: {test_acc}")

# 4.6 学習曲線のプロット
plt.plot(train_acc_list, label="train acc")
plt.plot(test_acc_list, label="test acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.savefig("train_test_acc.png")
