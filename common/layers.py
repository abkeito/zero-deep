# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum =momentum
        self.input_shape = None # Conv層の場合は4次元

        # テスト時の平均・分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時の変数
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        
        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

        else:
            xc = x - self.running_mean
            std = np.sqrt(self.running_var + 1e-7)
            xn = xc / std
        
        out  = self.gamma * xn + self.beta
        return out
    
    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx
    
    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xc * dout, axis =0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = - np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx

class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
    

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W= W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ (backward時に必要)
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad) # (データ数 * 出力の高さ * 出力の幅、チャンネル数 * フィルタの高さ * フィルタの幅)
        col_W = self.W.reshape(FN, -1).T # (チャンネル数 * フィルタの高さ * フィルタの幅、フィルタ数)
        out = np.dot(col, col_W) + self.b # (データ数 * 出力の高さ * 出力の幅、フィルタ数)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # (データ数、フィルタ数、出力の高さ、出力の幅)に変形

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)  #  (データ数 * 出力の高さ * 出力の幅、フィルタ数)

        self.db = np.sum(dout, axis=0) # (フィルタ数)
        self.dW = np.dot(self.col.T, dout) # (チャンネル数 * フィルタの高さ * フィルタの幅、フィルタ数)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW) # (フィルタ数、チャンネル数、フィルタの高さ、フィルタの幅)dx = col2im

        dcol = np.dot(dout, self.col_W.T) # (データ数 * 出力の高さ * 出力の幅、チャンネル数 * フィルタの高さ * フィルタの幅)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx
    
class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) # (データ数 * 出力の高さ * 出力の幅、チャンネル数 * フィルタの高さ * フィルタの幅)
        col = col.reshape(-1, self.pool_h * self.pool_w) # (データ数 * 出力の高さ * 出力の幅 * チャンネル数、フィルタの高さ * フィルタの幅)

        arg_max = np.argmax(col, axis=1) # (データ数 * 出力の高さ * 出力の幅 * チャンネル数)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # (データ数、フィルタ数、出力の高さ、出力の幅)に変形

        self.x = x
        self.arg_max = arg_max
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1) # (データ数、出力の高さ、出力の幅、フィルタ数)
        pool_size= self.pool_h * self.pool_w
        # 各プーリング領域に対して、全要素の勾配を仮で用意
        dmax = np.zeros((dout.size, pool_size)) # (データ数 * 出力の高さ * 出力の幅 * フィルタ数、フィルタの高さ * フィルタの幅)
        # 最大値が入っていたところだけに勾配を入れる
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten() # (データ数 * 出力の高さ * 出力の幅 * フィルタ数)
        # dmaxを元の形に戻す -> (データ数、出力の高さ、出力の幅、フィルタ数、フィルタの高さ * フィルタの幅)
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        # dmaxを(データ数*出力の高さ*出力の幅、フィルタ数*フィルタの高さ*フィルタの幅)に変形
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # dcolを(データ数、チャネル数、入力の高さ、入力の幅)に変形
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

