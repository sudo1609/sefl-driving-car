import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

data_dir = 'data'
data = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
data_image = data[['center', 'left', 'right']].values
steering = data['steering'].values

images = np.empty(32,)
images[4] = 5
print(images)