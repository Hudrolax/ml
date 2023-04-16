from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import Adam
from keras import layers
from keras.utils import Sequence
import xarray as xr
import numpy as np
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))

class CustomDataGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X, self.y = X, y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array(batch_X), np.array(batch_y)


print('load dataset')
dataset = xr.open_dataarray('dataset.nc')

# Разделение данных на обучающую и тестовую выборки
print('split data')
split_index = int(0.8 * dataset.shape[0])
train_data = dataset[:split_index].to_numpy()[:, :, :, None]
test_data = dataset[split_index:].to_numpy()[:, :, :, None]

print('make generator')
data_generator = CustomDataGenerator(train_data, train_data, 1000)
validation_generator = CustomDataGenerator(test_data, test_data, 200)

print('make model')
# Создание модели автоэнкодера на основе Conv1D
input_layer = layers.Input(shape=(100, 144, 1))
encoder = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
encoder = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(encoder)
encoder = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
encoder = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(encoder)
encoder = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
encoder_output = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(encoder)

# Декодер
decoder = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(encoder_output)
decoder = layers.UpSampling2D(size=(2, 2))(decoder)
decoder = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(decoder)
decoder = layers.UpSampling2D(size=(2, 2))(decoder)
decoder = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(decoder)
decoder = layers.UpSampling2D(size=(2, 2))(decoder)
decoder_output = layers.Conv2D(filters=1, kernel_size=(3, 3), activation='linear', padding='same')(decoder)
# Обрезание выхода декодера до нужного размера
decoder_output = layers.Cropping2D(cropping=((2, 2), (0, 0)))(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder_output)

# Компиляция и обучение модели
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
print('fit the model')
autoencoder.fit(data_generator,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=validation_generator)
# autoencoder.fit(train_data, train_data,
#                 epochs=20,
#                 batch_size=32,
#                 shuffle=True,
#                 validation_data=(test_data, test_data))

autoencoder.evaluate(test_data, test_data)
autoencoder.save('best_model/autoencoder2d.h5')
print('model saved')