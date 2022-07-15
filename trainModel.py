import pull
import getData
import os
import time
import numpy as np
from tensorflow import keras
from keras.layers import Dropout

class createModel():

	def __init__(self, dense, denseUnits ,inputShape, interval, timesteps,learning_rate,dropout):#add learning rate maybe? 3d_32du_24hi_30ts with lr 0.0001 and 0.1 dropout pretty good no sign of overfitting at 400 epochs. 
		#2d_32du_24hi_30ts_0.00005lr and no drop out best so far with 66% accuarcy

		self.name = f"{dense}d_{denseUnits}du_{interval}i_{timesteps}ts_{time.time()}"

		#if os.path.exists(self.name):

			#self.model = keras.models.load_model(self.name)

		#else:

		self.model = keras.Sequential()
		
		for i in range(dense-1):
			self.model.add(keras.layers.Dense(denseUnits, activation = "relu"))
			#self.model.add(Dropout(dropout))

		self.model.add(keras.layers.Dense(denseUnits, activation = "relu"))

		self.model.add(keras.layers.Dense(1, activation = "sigmoid"))

		optimizer = keras.optimizers.Adam(0.001)
		optimizer.learning_rate.assign(learning_rate)
		self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics="binary_accuracy")
		#metrics will always be the same for classification problems so i didnt include them as parameters

	def train(self, X, Y, validationSplit, batchsize, epochs):
		tb = keras.callbacks.TensorBoard(log_dir = f"logs/{self.name}")
		self.model.fit(X, Y, batch_size = batchsize, epochs= epochs, callbacks = [tb], validation_split = validationSplit)
		self.model.save(self.name)

#func that uses updater to get data and passes it to create model object that trains model using data. Gonna have to do some bugfixing my dude.
def modelforID(interval,iden,timesteps):
	updater= getData.Updater(interval,iden[0])
	data,labels = updater.processData(timesteps)

	for i in iden[1:]: #include all the different item ids you want to save
		updater= getData.Updater(interval,i)
		tempa,tempb = updater.processData(timesteps)
		data  = np.concatenate((data,tempa))
		labels  = np.concatenate((labels,tempb))

	data,labels = getData.randomiseInUnison(data,labels)
	#print (data,labels)

	data = getData.reshapeData(data)

	model = createModel(2,32,data.shape[1:],interval,timesteps,0.00005,0.1) # input shape must be samples,number of time steps in each sample,number of features in each timestep
	model.train(data,labels,0.1,5,2000)

if __name__ == "__main__":
	modelforID("24h",[22481,22324,12924,22486,20997,22978,8921,4151,13263],30)