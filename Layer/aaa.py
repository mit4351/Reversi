# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(xx_train, tt_train), (x_test, t_test) = load_mnist(normalize=True)

# 過学習を再現するために、学習データを削減
#x_train = xx_train[:300]
#t_train = xx_train[:300]
#バリデーションデータ
validation_cnt=10000
x_valid = xx_train[:validation_cnt]
t_valid = tt_train[:validation_cnt]
#トレーニングデータ
x_train = xx_train[validation_cnt:]
t_train = tt_train[validation_cnt:]

use_dropout = True  # Dropoutなしのときの場合はFalseに
dropout_ratio = 0.15
file_name="aaa.pkl"

if os.path.isfile(file_name) :
    with open(file_name, 'rb') as f:
        network  =   pickle.load(f)
else:
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=60, mini_batch_size=1000,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()

with open(file_name, 'wb') as f:
    pickle.dump(network, f)

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# グラフの描画==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
