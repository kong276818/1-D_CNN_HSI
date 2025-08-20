# train_1dcnn_bgfilter.py
import os, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.models import Model
from os.path import join
from sklearn.model_selection import train_test_split

# -------------------
# 경로/데이터 나누기
# -------------------
np_path = './npy_label'
np_files = sorted(os.listdir(np_path))

data_files  = [f for f in np_files if f.endswith('.npy') and 'labels' not in f]
label_files = [f for f in np_files if f.endswith('.npy') and 'labels' in f]
file_pairs  = list(zip(data_files, label_files))
file_pairs  = [(join(np_path, d), join(np_path, l)) for d, l in file_pairs]

train_files, valtest_files = train_test_split(file_pairs, test_size=0.3, random_state=100, shuffle=True)
val_files,   test_files    = train_test_split(valtest_files, test_size=0.4, random_state=100, shuffle=True)

# -------------------
# 배경(왼쪽 검정) 제거 유틸
# -------------------
INTENSITY_PCTL     = 95    # 밝기 프록시 생성용 퍼센타일
DARK_THR_RATIO     = 0.02  # 0~1 스케일에서 '완전 검정' 임계
MIN_FORE_COL_RATIO = 0.05  # 한 컬럼의 '밝은 픽셀' 비율이 이 값 넘으면 전경 시작

TARGET_BANDS = 102
TRIM_FRONT   = 2  # 밴드가 104라면 2~103 사용

def left_bg_mask_and_flatten(cube, labels):
    """
    cube: (H,W,B), labels: (H*W,)
    왼쪽 검정 배경만 배제하여 foreground만 (N,B)로 반환, 라벨도 동기화
    """
    H, W, B = cube.shape
    cube_f = cube.astype(np.float32, copy=False)

    inten = np.percentile(cube_f, INTENSITY_PCTL, axis=2)
    vmin, vmax = float(inten.min()), float(inten.max())
    inten_norm = (inten - vmin) / (max(vmax - vmin, 1e-12))

    bright = inten_norm > DARK_THR_RATIO
    ratio_col = bright.mean(axis=0)
    fg_start = W
    for j in range(W):
        if ratio_col[j] >= MIN_FORE_COL_RATIO:
            fg_start = j
            break

    # 배경: 왼쪽 구간에서 어두운 픽셀만
    bg_mask = np.zeros((H, W), dtype=bool)
    if fg_start > 0:
        bg_mask[:, :fg_start] = ~bright[:, :fg_start]

    fg_mask = ~bg_mask  # 학습에 쓸 전경만 남김
    cube_flat = cube_f.reshape(H*W, B)
    labels    = labels.reshape(-1)
    fg_idx    = np.where(fg_mask.reshape(-1))[0]

    if fg_idx.size == 0:
        # 전경이 아예 없으면 전체 사용(안전장치)
        fg_idx = np.arange(H*W)

    X = cube_flat[fg_idx]
    y = labels[fg_idx]
    return X, y

def filter_too_dark_rows(X, y):
    """
    입력이 (N,B)일 때: 너무 어두운 스펙트럼(배경)을 제거
    """
    # 밝기 프록시: 각 스펙트럼의 pctl
    p = np.percentile(X, INTENSITY_PCTL, axis=1)
    # 0~1 정규화용 전역 min/max
    pmin, pmax = float(p.min()), float(p.max())
    if pmax <= pmin:
        return X, y  # 전부 동일하면 필터 못함
    pnorm = (p - pmin) / (pmax - pmin)
    keep = pnorm > DARK_THR_RATIO
    if keep.sum() == 0:
        return X, y
    return X[keep], y[keep]

def fit_to_target_bands(arr_2d):
    """
    (N,B) -> (N, TARGET_BANDS)
    """
    N, B = arr_2d.shape
    if B >= TARGET_BANDS + TRIM_FRONT:
        arr_2d = arr_2d[:, TRIM_FRONT:TRIM_FRONT + TARGET_BANDS]
    elif B >= TARGET_BANDS:
        arr_2d = arr_2d[:, B - TARGET_BANDS:]
    else:
        pad = TARGET_BANDS - B
        arr_2d = np.pad(arr_2d, ((0,0),(0,pad)), mode='edge')
    return arr_2d

def preprocess_X(X):
    """
    (N,B) -> gradient -> (N, TARGET_BANDS, 1)
    """
    X = np.gradient(X, axis=1).astype(np.float32)
    X = fit_to_target_bands(X)
    return np.expand_dims(X, axis=-1)

# -------------------
# 제너레이터
# -------------------
def npy_generator(file_list):
    for data_path, label_path in file_list:
        X = np.load(data_path)   # (H,W,B) 또는 (N,B)
        y = np.load(label_path)  # (N,)

        if X.ndim == 3:
            # 왼쪽 검정 배경 제거 + 전경만 평탄화
            X, y = left_bg_mask_and_flatten(X, y)
        elif X.ndim == 2:
            # (N,B)일 때: 너무 어두운 row 제거
            X, y = filter_too_dark_rows(X, y)
        else:
            raise ValueError(f"Unsupported X shape: {X.shape}")

        # 전처리
        X = preprocess_X(X)  # (N, TARGET_BANDS, 1)

        for i in range(X.shape[0]):
            yield X[i], np.int32(y[i])

def create_dataset(file_list, batch_size=512, shuffle=False):
    output_signature = (
        tf.TensorSpec(shape=(TARGET_BANDS, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(lambda: npy_generator(file_list),
                                        output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(4096, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# -------------------
# 데이터셋
# -------------------
batch_size = 256  # 안정화를 위해 축소
train_ds = create_dataset(train_files, batch_size=batch_size, shuffle=True)
val_ds   = create_dataset(val_files,   batch_size=batch_size, shuffle=False)
test_ds  = create_dataset(test_files,  batch_size=batch_size, shuffle=False)

# -------------------
# 모델 정의 (1D CNN)
# -------------------
input_shape = (TARGET_BANDS, 1)
inputs = Input(shape=input_shape, name='conv1d_input')
x = Conv1D(64, 9, strides=1, activation='relu', padding='same')(inputs)
x = Conv1D(64, 9, strides=2, activation='relu', padding='same')(x)
x = Dropout(0.2)(x)
x = Conv1D(128, 7, strides=1, activation='relu', padding='same')(x)
x = Conv1D(128, 7, strides=2, activation='relu', padding='same')(x)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(11, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.Adam(learning_rate=1e-4)  # ↓ 안정화
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

# -------------------
# 학습
# -------------------
epochs = 20
model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)

# -------------------
# 저장 및 평가
# -------------------
os.makedirs('./models', exist_ok=True)
model.save('./models/model.h5')

loss, acc = model.evaluate(test_ds, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))

now = time.localtime()
save_path = join('./models/', f"{now.tm_year}_{now.tm_mon}_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}")
model.save(save_path)
