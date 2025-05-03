import os, sys
import numpy as np
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートできるようにする
import matplotlib.pyplot as plt
from dataset_zero.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD, Momentum, AdaGrad, Adam

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
train_size = x_train.shape[0]
batch_size = 128
max_iter = 2000

# 実験の設定
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                   output_size=10)
    train_loss[key] = []

# 学習
for i in range(max_iter):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("========" + "iteration: " + str(i) + "========")
        for key in optimizers.keys():
            loss == networks[key].loss(x_batch, t_batch)
            print(key + " : " + str(loss))

# グラフの描画
markers = {'SGD': 'o', 'Momentum': 'x', 'AdaGrad': 's', 'Adam': 'D'}
x = np.arange(max_iter)
for key in optimizers.keys():
    plt.plot(x, train_loss[key], marker=markers[key], label=key, markevery=100, markersize=0.1)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.savefig("optimizer_compare_mnist.png")

