Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. Create a convolutional neural network that trains to 100% accuracy on these images, which cancels training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.

import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
​
# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/happy-or-sad.zip"
​
zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()
# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.
​
    DESIRED_ACCURACY = 0.999
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('acc') > DESIRED_ACCURACY):
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training=True
        
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        # Your Code Here
        tf.keras.layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation=tf.nn.relu),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    
    
​
    from tensorflow.keras.optimizers import RMSprop
​
    model.compile(
                  loss='binary_crossentropy',
                  optimizer=RMSprop(lr = 0.001),
                  metrics=['acc']
        
    )
    
   
        
    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory
​
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
​
    train_datagen = ImageDataGenerator(rescale=1./255)
                    
    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        # Your Code Here
    "/tmp/h-or-s",
    target_size=(150,150),
    batch_size=4,
    class_mode='binary'
    )
    
    
    callbacks=myCallback()
    
    history = model.fit_generator(
        # Your Code Here
        train_generator,
        steps_per_epoch=2,
        epochs=18,
        verbose=1,
        callbacks=[callbacks])
     # model fitting
    return history.history['acc'][-1]
# The Expected output: "Reached 99.9% accuracy so cancelling training!""
train_happy_sad_model()
WARNING: Logging before flag parsing goes to stderr.
W0622 01:56:21.299214 140118618986304 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
W0622 01:56:21.707204 140118618986304 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Found 80 images belonging to 2 classes.
Epoch 1/18
2/2 [==============================] - 5s 2s/step - loss: 5.7566 - acc: 0.3750
Epoch 2/18
2/2 [==============================] - 0s 5ms/step - loss: 0.9629 - acc: 0.6250
Epoch 3/18
2/2 [==============================] - 0s 47ms/step - loss: 0.6619 - acc: 0.7500
Epoch 4/18
2/2 [==============================] - 0s 48ms/step - loss: 0.9454 - acc: 0.6250
Epoch 5/18
2/2 [==============================] - 0s 48ms/step - loss: 0.6626 - acc: 0.6250
Epoch 6/18
2/2 [==============================] - 0s 6ms/step - loss: 0.6487 - acc: 0.7500
Epoch 7/18
2/2 [==============================] - 0s 44ms/step - loss: 0.4382 - acc: 0.8750
Epoch 8/18
2/2 [==============================] - 0s 5ms/step - loss: 0.6215 - acc: 0.6250
Epoch 9/18
2/2 [==============================] - 0s 42ms/step - loss: 0.6779 - acc: 0.5000
Epoch 10/18
2/2 [==============================] - 0s 5ms/step - loss: 0.5135 - acc: 0.7500
Epoch 11/18
1/2 [==============>...............] - ETA: 0s - loss: 0.2715 - acc: 1.0000
Reached 99.9% accuracy so cancelling training!
2/2 [==============================] - 0s 8ms/step - loss: 0.2050 - acc: 1.0000
1.0
# Now click the 'Submit Assignment' button above.
# Once that is complete, please run the following two cells to save your work and close the notebook
%%javascript
<!-- Save the notebook -->
IPython.notebook.save_checkpoint();
%%javascript
IPython.notebook.session.delete();
window.onbeforeunload = null
setTimeout(function() { window.close(); }, 1000);
