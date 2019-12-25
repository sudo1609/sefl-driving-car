import pandas as pd
import numpy as np
import cv2
import os
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Lambda
from keras.callbacks import ModelCheckpoint
from augmentation import INPUT_SHAPE, batch_generator
from keras.optimizers import Adam

data_dir = 'data'
data = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
data_image = data[['center', 'left', 'right']].values
steering = data['steering'].values

#before process
#plt.hist(steering)

#loai bo chi lay 1000 anh co steering = 0. Cach 1
zero = []
non_zero = []
for i in range(len(steering)):
    if steering[i] == 0:
        zero.append(i)
    else:
        non_zero.append(i)
zero = np.array(zero)
non_zero = np.array(non_zero)
np.random.shuffle(zero)
zero = zero[:1000]
processed = np.hstack((zero, non_zero))
steering = steering[processed].reshape(len(processed))
data_image = data_image[processed, :].reshape(len(processed), -1)

#after process
#plt.hist(steering)
#plt.show()

#Cach 2:
#zero = np.array(np.where(steering==0)).reshape(-1, 1)
#none_zero = np.array(np.where(steering!=0)).reshape(-1, 1)
#np.random.shuffle(zero)
#zero = zero[:1000]
#pos_combined = np.vstack((zero, none_zero))
#steering = steering[pos_combined].reshape(len(pos_combined))
#data_image = data_image[pos_combined, :].reshape(len(pos_combined), 3)

# Chia ra traing set và validation set
X_train, X_valid, y_train, y_valid = train_test_split(data_image, steering, test_size=0.2, \
random_state=0)

#xây dựng model
model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()


batch_size = 32
Learning_rate = 1e-4
Save_best_only = True
file_Path = 'best_weight-{epoch:03d}.h5'
epoch = 10
samples_epoch = 1000

#checkpoint lưu lại giá trị weight nếu như model đại giá trị loss thấp nhất
checkpoint = ModelCheckpoint(file_Path,
                             monitor = 'val_loss',
                             verbose = 1,
                             save_best_only = Save_best_only,
                             mode='auto')

#dùng mean_squared_error làm hàm loss_function để lấy giá trị thực.
model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate = Learning_rate))

History = model.fit_generator(batch_generator(data_dir, X_train, y_train, batch_size, True),
steps_per_epoch = samples_epoch,
epochs = epoch,
max_q_size=1,
validation_data=batch_generator(data_dir, X_valid, y_valid, \
batch_size, False),
nb_val_samples=len(X_valid),
callbacks=[checkpoint],
verbose=1)


