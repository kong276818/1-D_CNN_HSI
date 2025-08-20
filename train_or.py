#train 1dcnn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np
import os
from os.path import join
import time
from sklearn.model_selection import train_test_split

np_path = './npy_label'
np_files = sorted(os.listdir(np_path))

# 데이터 파일을 분류
data_files = [f for f in np_files if 'labels' not in f]
label_files = [f for f in np_files if 'labels' in f]

# 데이터셋 생성용 파일 목록 생성
file_pairs = list(zip(data_files, label_files))

# 전체 파일 경로화
file_pairs = [(join(np_path, d), join(np_path, l)) for d, l in file_pairs]

# 학습/검증/테스트 나누기
train_files, valtest_files = train_test_split(file_pairs, test_size=0.3, random_state=100)
val_files, test_files = train_test_split(valtest_files, test_size=0.4, random_state=100)

def npy_generator(file_list):
    for data_path, label_path in file_list:
        data = np.load(data_path)
        labels = np.load(label_path)
        data = np.gradient(data, axis=1).astype(np.float32)
        if data.shape[1] > 102:
            data = data[:, 2:]
        for i in range(data.shape[0]):
            yield np.expand_dims(data[i], axis=-1), np.int32(labels[i])

def create_dataset(file_list, batch_size=1024, shuffle=False):
    output_signature = (
        tf.TensorSpec(shape=(102, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    ds = tf.data.Dataset.from_generator(
        lambda: npy_generator(file_list),
        output_signature=output_signature
    )
    if shuffle:
        ds = ds.shuffle(1000)
    return ds.batch(batch_size)

# 데이터셋 생성

batch_size = 128
train_ds = create_dataset(train_files, batch_size=batch_size, shuffle=True)
val_ds = create_dataset(val_files, batch_size=batch_size)
test_ds = create_dataset(test_files, batch_size=batch_size)

# 모델 정의
input_shape = (102, 1)
inputs = Input(shape=input_shape, name='conv1d_input')
x = Conv1D(48, 24, strides=2, activation='relu')(inputs)
x = Conv1D(32, 24, strides=2, activation='relu')(x)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(8, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

# 학습
epoch = 50
model.fit(train_ds, epochs=epoch, validation_data=val_ds, verbose=1)

# 저장
model.save('./models/model.h5')

loss, acc = model.evaluate(test_ds, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))

now = time.localtime()
save_path = join('./models/', f"{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}")
model.save(save_path)