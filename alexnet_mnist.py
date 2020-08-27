#importing packages
import tensorflow as tf 
import numpy as np 

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=np.array(x_train)
y_train=np.array(y_train)

x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1) 

class AlexNet:
	def __init__(self,x,y):
		self.x=x
		self.y=y
		self.model=tf.keras.models.Sequential()

	def call(self):
		self.model.add(tf.keras.layers.Conv2D(filters=96,kernel_size=11,strides=1,padding="valid"))
		self.model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
		self.model.add(tf.keras.layers.ReLU())

		self.model.add(tf.keras.layers.ZeroPadding2D(2,2))
		self.model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=5,strides=1,padding="valid"))
		self.model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
		self.model.add(tf.keras.layers.ReLU())

		self.model.add(tf.keras.layers.ZeroPadding2D(1,1))
		self.model.add(tf.keras.layers.Conv2D(filters=384,kernel_size=3,strides=1,padding="valid"))
		self.model.add(tf.keras.layers.ReLU())

		self.model.add(tf.keras.layers.ZeroPadding2D(1,1))
		self.model.add(tf.keras.layers.Conv2D(filters=384,,kernel_size=3,strides=1,padding="valid"))
		self.model.add(tf.keras.layers.ReLU())

		self.model.add(tf.keras.layers.ZeroPadding2D(1,1))
		self.model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding="valid"))
		self.model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
		self.model.add(tf.keras.layers.ReLU())

		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dense(unit=4096,activation="relu"))
		self.model.add(tf.keras.layers.Dense(unit=10,activation="softmax"))
		self.model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
		self.model.fit(self.x,self.y,batch_size=100,steps_per_epoch=1,epochs=1)
emp=LeNet(x=x_train,y=y_train).call()
		




