# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError("Python 3 is required to run this script.")
import os.path
import gzip
import pickle
import os
import numpy as np

url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # mirror site

key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = os.path.join(dataset_dir, 'mnist.pkl')

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _download(file_name):
    file_path = os.path.join(dataset_dir, file_name)

    if os.path.exists(file_path):
        return
    
    print(f'Downloading {file_name}...')
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    request = urllib.request.Request(url_base + file_name, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(file_path, 'wb') as f:
        f.write(response)
    print('Done.')

def download_mnist():
    for v in key_file.values():
        _download(v)

def _load_label(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    print("Converting", file_name, "to numpy array...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done.")
    return labels

def _load_img(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    print("Converting", file_name, "to numpy array...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done.")
    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done.")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNISTデータセットの読み込み
    Parameters
    ----------
    normalize : bool
        Trueの場合、画素値を0.0~1.0に正規化する
    flatten : bool
        Trueの場合、28x28の画像を784次元のベクトルに変換する
    one_hot_label : bool
        Trueの場合、ラベルをone-hotベクトルに変換する
    
    Returns
    -------
    (train_img, train_label), (test_img, test_label)
        train_img : 訓練画像
        train_label : 訓練ラベル
        test_img : テスト画像
        test_label : テストラベル
    """
    if not os.path.exists(save_file):
        init_mnist()
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

