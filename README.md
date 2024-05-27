# ANN_Mnist_Tensorflow
This Python notebook demonstrates the process of building and training a simple neural network using TensorFlow and Keras on the MNIST dataset. Below is a step-by-step description of the notebook's contents and functionality.

##Prerequisites
</Ensure that the required libraries are installed. This notebook specifically uses patchify and tensorflow-gpu.>


`!pip install patchify
!pip install tensorflow-gpu`

##Import Libraries
</The necessary libraries are imported, including TensorFlow, numpy, matplotlib, and others.>


`import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt`

##Check GPU Availability
The notebook checks for GPU availability and verifies its usage.

`my_gpu_device_name = tf.test.gpu_device_name()
if my_gpu_device_name != '/device:GPU:0':
    raise SystemError('GPU not found!')
print(f'Available GPU : {my_gpu_device_name}')`

Additionally, the specific GPU details and memory usage are displayed.
`gpu_info = !nvidia-smi
print('\n'.join(gpu_info))`

##Check RAM Availability
The total available RAM is checked to ensure the runtime is sufficient for training.
`from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))`

##TensorFlow Version
The version of TensorFlow used is printed for reference.
`print(tf.__version__)`

#Load MNIST Dataset
</The MNIST dataset is loaded, split into training and testing sets, and visualized.>
`mnist = datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()`

</Visualize a few examples>
plt.figure(figsize=(10, 2))
for index, (image, label) in enumerate(zip(x_train[0:5], y_train[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(f'Training: {label}', fontsize=10)
    
##Data Preprocessing
</The data is normalized by converting the pixel values from integers to floating-point numbers between 0 and 1.>
x_train, x_test = x_train / 255.0, x_test / 255.0

##Build the Model
</A sequential model is built with a single hidden layer.>
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

##Compile the Model
</The model is compiled with an optimizer, loss function, and evaluation metrics.>
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
##Train the Model
</The model is trained on the training data for 50 epochs.>
model.fit(x_train, y_train, epochs=50)

##Evaluate the Model
</The model's performance is evaluated on the test dataset.>
loss, accuracy = model.evaluate(x_test, y_test)
print(f"loss = {loss}, accuracy = {accuracy}")

##Predictions
</Predictions are made on the test dataset, and the predicted classes are extracted.>
predictions = model.predict(x_test)
classes = np.argmax(predictions, axis=1)

##Visualize Predictions
</A few test images along with their predicted labels are visualized.>
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(x_test[0:10], classes[0:10])):
    plt.subplot(2, 5, index + 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(f'Training: {label}', fontsize=10)
    
##Confusion Matrix
</A confusion matrix is plotted to visualize the performance of the model.>
from sklearn import metrics
import seaborn as sns
cm = metrics.confusion_matrix(y_test, classes, normalize='true')
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title(f'Accuracy = {accuracy}', size=15)

##Callbacks for Early Stopping and Model Checkpoint
</Callbacks are implemented to monitor the training process, including early stopping and model checkpointing.>
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
import os, datetime

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tnsrboard = TensorBoard(logdir, histogram_freq=1)
csv_logger = CSVLogger("/content/logs.csv", append=True, separator=',')

callbacks = [early_stopping, model_checkpoint, tnsrboard, csv_logger]

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    validation_split=0.25,
    batch_size=40,
    verbose=2,
    callbacks=[callbacks]
)

##Load and Evaluate Saved Model
</The best model saved during training can be loaded and evaluated on the test dataset.>
from keras.models import load_model
saved_model = load_model('best_model.h5')
test_acc = saved_model.evaluate(x_test, y_test)

##TensorBoard
</TensorBoard is used for visualizing the training process.>
%load_ext tensorboard
%tensorboard --logdir logs
This notebook provides a comprehensive guide to building, training, and evaluating a simple neural network using TensorFlow and Keras on the MNIST dataset. It also demonstrates the use of callbacks for efficient training and model management.
