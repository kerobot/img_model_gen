import sys
import os
import random
import pathlib
import shutil
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

print("学習データのラベル付け開始")

img_file_name_list=os.listdir("./face_scratch_image/")
print(len(img_file_name_list))

for i in range(0,len(img_file_name_list)):
    n=os.path.join("./face_scratch_image",img_file_name_list[i])
    img = cv2.imread(n)
    if isinstance(img,type(None)) == True:
        img_file_name_list.pop(i)
        continue
print(len(img_file_name_list))

X_train=[]
y_train=[]
for j in range(0,len(img_file_name_list)):
    n=os.path.join("./face_scratch_image/",img_file_name_list[j])
    img = cv2.imread(n)
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    X_train.append(img)
    n=img_file_name_list[j]
    y_train=np.append(y_train,int(n[0:2])).reshape(j+1,1)
X_train=np.array(X_train)

print("学習データのラベル付け終了")

print("テストデータのラベル付け開始")

img_file_name_list=os.listdir("./test_image/")
print(len(img_file_name_list))

for i in range(0,len(img_file_name_list)):
    n=os.path.join("./test_image",img_file_name_list[i])
    img = cv2.imread(n)
    if isinstance(img,type(None)) == True:
        img_file_name_list.pop(i)
        continue
print(len(img_file_name_list))

X_test=[]
y_test=[]
for j in range(0,len(img_file_name_list)):
    n=os.path.join("./test_image",img_file_name_list[j])
    img = cv2.imread(n)
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    X_test.append(img)
    n=img_file_name_list[j]
    y_test=np.append(y_test,int(n[0:2])).reshape(j+1,1)
X_test=np.array(X_test)

print("テストデータのラベル付け終了")

print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
print("X_test.shape:", X_test.shape)
print("y_test.shape:", y_test.shape)

from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard

# 特徴量の正規化
# X_train = X_train / 255.
# X_test = X_test / 255.

# クラスラベルの1-hotベクトル化
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# plt.imshow(X_train[0])
# plt.show()
# print(y_train[0])

# モデルの定義
model = Sequential()

# フィルタを用いてストライド1で特徴マップを計算
# input_shape   入力データのサイズ 64 x 64 x RGB
# filters       出力チャンネル数
# kernel_size   フィルタ(=カーネル)のサイズ数．3x3とか5x5とか奇数正方にすることが一般的
# strides       ストライドの幅(フィルタを動かすピクセル数)
# padding       データの端の取り扱い方(入力データの周囲を0で埋める(ゼロパディング)ときは'same',ゼロパディングしないときは'valid')
# activation    活性化関数
model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
# 2x2の4つの領域に分割して各2x2の行列の最大値をとる
# データが縮小されることで計算コストが軽減される+各領域内の位置の違いを無視するためモデルが小さな位置変化に対して頑健となる
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# ドロップアウト層
model.add(Dropout(0.01))

# 全結合層(プーリング層の出力は4次元テンソルであるため2次元テンソルに展開)
model.add(Flatten())

model.add(Dense(256))
model.add(Activation('sigmoid'))

model.add(Dense(128))
model.add(Activation('sigmoid'))

model.add(Dense(2))
model.add(Activation('softmax'))

# コンパイル
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

# 学習
# model.fit(X_train, y_train, batch_size=32, epochs=60)

# グラフ用
history = model.fit(X_train, y_train, batch_size=32, epochs=60, verbose=1, validation_data=(X_test, y_test))

# 汎化制度の評価・表示
score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

# acc, val_accのプロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

# モデルを保存
model.save("model.h5")
