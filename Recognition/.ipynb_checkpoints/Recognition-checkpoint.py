# 導入函式庫
from preprocess_v2 import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np

# save_data_to_array(path="C:\\Users\\NDHU\\Desktop\\英文單字辯識\\SpeechRecognition\\data\\1dollarmoney", max_pad_len=11)
save_data_to_array(path='./data/1dollarmoney', label=0, max_pad_len=15)
save_data_to_array(path='./data/10dollarmoney', label=1, max_pad_len=15)
save_data_to_array(path='./data/50dollarmoney', label=2, max_pad_len=15)

# The list order defines the labels
filepath_list = [
    './1dollarmoney.npy', # label: 0
    './10dollarmoney.npy', # label: 1
    './50dollarmoney.npy'  # label: 2
]

# 載入 data 資料夾的訓練資料，並自動分為『訓練組』及『測試組』
X_train, X_test, y_train, y_test = get_train_test(filepath_list)
X_train = X_train.reshape(X_train.shape[0], 20, 15, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 15, 1)

# 類別變數轉為one-hot encoding
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
print("X_train.shape=", X_train.shape)

# 建立簡單的線性執行的模型
model = Sequential()
# 建立卷積層，filter=32,即 output size, Kernal Size: 2x2, activation function 採用 relu
model.add(Conv2D(100, kernel_size=(1, 1), activation='relu', input_shape=(20, 15, 1)))
# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling2D(pool_size=(1, 1)))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
model.add(Dropout(0.25))
# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
model.add(Flatten())
# 全連接層: 128個output
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
# Add output layer
model.add(Dense(3, activation='softmax'))
# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 進行訓練, 訓練過程會存在 train_history 變數中
model.fit(X_train, y_train_hot, batch_size=16, epochs=200, verbose=1, validation_data=(X_test, y_test_hot))

# 評估模型
score = model.evaluate(X_test, y_test_hot, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# 預測(prediction)
mfcc = wav2mfcc(r"C:\Users\王天佑\Desktop\1-002.wav",15)
mfcc_reshaped = mfcc.reshape(1, 20, 15, 1)
print("labels=0 is mean 1 dollarmoney ,1 is mean 10 dollar money ,2 is mean 50dollar money")
print("predict=", np.argmax(model.predict(mfcc_reshaped)))

# 混淆矩陣
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(np.argmax(y_test_hot, axis=1), y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)
