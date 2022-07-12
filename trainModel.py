import pull
import getData
import os
import numpy as np
from tensorflow import keras
from keras.layers import Dropout

class createModel():

	def __init__(self, dense, denseUnits ,inputShape, outputUnits, interval, timesteps):#add learning rate maybe?

		self.name = f"{dense}d_{denseUnits}du_{interval}i_{timesteps}ts"

		if os.path.exists(self.name):

			self.model = keras.models.load_model(self.name)

		else:

			self.model = keras.Sequential()
			
			for i in range(dense):
				self.model.add(keras.layers.Dense(denseUnits, activation = "relu"))
				self.model.add(Dropout(0.5))

			self.model.add(keras.layers.Dense(outputUnits, activation = "softmax"))

			self.model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
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

	data = getData.reshapeData(data)

	model = createModel(10,32,data.shape[1:],2,interval,timesteps) # input shape must be samples,number of time steps in each sample,number of features in each timestep
	model.train(data,labels,0.05,1,70)

if __name__ == "__main__":
	modelforID("5m",[449,4151],30)