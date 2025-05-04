import numpy as np

def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data: (データ数、チャンネル数、高さ、幅)の4次元配列からなる入力データ
    filter_h: フィルタの高さ
    filter_w: フィルタの幅
    stride; ストライド
    pad: パディング

    Returns
    -------
    col: 二次元配列 
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant') # 0で高さと幅にパディング
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) #  データ数、チャンネル数、フィルタの高さ、フィルタの幅、出力の高さ、出力の幅の6次元配列

    for y in range(filter_h): # フィルタのy方向を全て走査
        y_max = y + stride * out_h # フィルタのy行がかかる、入力のy行の最大値
        for x in range(filter_w): # フィルタのx方向を全て走査
            x_max = x + stride * out_w # フィルタのx行がかかる、入力のx行の最大値
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride] # フィルタのy行、x行がかかる、入力のy行、x行を入れてる。出力の高さと幅の分はブロードキャスト。
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    # transpose: (データ数、出力の高さ、出力の幅、チャンネル数、フィルタの高さ、フィルタの幅)に順番を変える
    # reshape: (データ数 * 出力の高さ * 出力の幅、チャンネル数 * フィルタの高さ * フィルタの幅)に変形
    # つまり、横向きに3次元フィルタ演算1回分のデータが入っていて、それをデータ数 * 出力の高さ * 出力の幅回繰り返すという構造
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :] # フィルタによって重なる部分を足し合わせる

    return img[:, :, pad:H + pad, pad:W + pad]