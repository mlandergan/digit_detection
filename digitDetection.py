import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

num_hidden_layers = 10
num_output = 10
num_nodes = 50
learning_rate = 0.01
num_batch_size = 32
num_epoch = 500

imgs =  np.load('/home/mlandergan/Documents/WPI/senior/AI/digitDetection/images.npy')
labels = np.load('/home/mlandergan/Documents/WPI/senior/AI/digitDetection/labels.npy')

train_imgs_size = imgs.shape[0]
train_labels_size = labels.shape[0]

imgs = imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2])
labels_binary = np_utils.to_categorical(labels, 10)

train_imgs = imgs[:int(train_imgs_size*(0.6))]
test_imgs = imgs[int(train_imgs_size*(.6)):int(train_imgs_size*(.8))]
validation_imgs = imgs[int(train_imgs_size*(.8)):]

train_labels = labels_binary[:int(train_labels_size*(.6))]
test_labels_binary = labels_binary[int(train_labels_size*(.6)):int(train_labels_size*(.8))]
test_labels = labels[int(train_imgs_size*(.6)):int(train_imgs_size*(.8))]
validation_labels = labels_binary[int(train_labels_size*(.8)):]

model = keras.Sequential()
model.add(keras.layers.Dense(num_nodes, input_dim=784, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2, noise_shape=None, seed=None))

# create the 10 hidden layers
for i in range(num_hidden_layers):
     model.add(keras.layers.Dense(num_nodes, activation=tf.nn.relu))

model.add(keras.layers.Dense(num_output, activation=tf.nn.softmax))

# Optimizer is stochastic gradient descent, loss function is mean squared error
sgd = keras.optimizers.SGD(lr=learning_rate)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

# train the model iterating on the data in batches of 512 samples
#history = model.fit(train_imgs, train_labels, verbose=1, validation_data=(validation_imgs, validation_labels), epochs =num_epoch, batch_size = num_batch_size)


#test_acc = model.evaluate(test_imgs, test_labels, verbose=1, batch_size = num_batch_size)
#print('Test accuracy:', test_acc)

def cross_valid(model, train_imgs, train_labels):

    accuracy = []

    img_size = train_imgs.shape[0]
    labels_size = train_labels.shape[0]

    train_imgs_set1 = train_imgs[:(img_size/3)]
    train_imgs_set2 = train_imgs[(img_size/3):(img_size*2/3)]
    train_imgs_set3 = train_imgs[(img_size*2/3):]

    train_labels_set1 = train_labels[:(labels_size/3)]
    train_labels_set2 = train_labels[(labels_size/3):(labels_size*2/3)]
    train_labels_set3 = train_labels[(labels_size*2/3):]

    # Fold 1
    history = model.fit(np.append(train_imgs_set1, train_imgs_set2, axis=0), np.append(train_labels_set1, train_labels_set2, axis=0), verbose=1, validation_data=(train_imgs_set3, train_labels_set3), epochs =num_epoch, batch_size = num_batch_size)
    print history.history
    accuracy.append(history.history['val_acc'])

    # Fold 2
    history = model.fit(np.append(train_imgs_set2, train_imgs_set3, axis=0), np.append(train_labels_set2, train_labels_set3, axis=0), verbose=1, validation_data=(train_imgs_set1, train_labels_set1), epochs =num_epoch, batch_size = num_batch_size)
    accuracy.append(history.history['val_acc'])

    # Fold 3
    history = model.fit(np.append(train_imgs_set1, train_imgs_set3, axis=0), np.append(train_labels_set1, train_labels_set3, axis=0), verbose=1, validation_data=(train_imgs_set2, train_labels_set2), epochs =num_epoch, batch_size = num_batch_size)
    accuracy.append(history.history['val_acc'])

    return accuracy

#accuracy = cross_valid(model, imgs, labels_binary)
accuracy = cross_valid(model, test_imgs, test_labels_binary)

#print 'Mean Accuracy: {}, {}, {}'.format(np.mean(accuracy[0]), np.mean(accuracy[1]), np.mean(accuracy[2]))

prediction = model.predict(test_imgs, batch_size = num_batch_size)
predicted_digits = np.argmax(prediction, axis=1)

"""
total_error = 0
for i in range(len(predicted_digits)):
     error = predicted_digits[i] - val_labels[i]
     total_error += error*error
total_error /= len(predicted_digits)
print(total_error)
"""
predicted_digits_pd = pd.Series(predicted_digits, name='Predicted')
actual_digits_pd = pd.Series(test_labels, name='Actual')
df_confusion = pd.crosstab(actual_digits_pd, predicted_digits_pd)
print df_confusion


plt.plot(accuracy[0])
plt.plot(accuracy[1])
plt.plot(accuracy[2])
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend(['Fold 1', 'Fold 2', 'Fold3'], loc='lower right')
plt.show()
