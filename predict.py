import getData
from tensorflow import keras

class predictor():

	def __init__(self,interval,timesteps):
		self.timesteps = timesteps
		self.interval = interval

	def getmodel(self, dense, denseUnits):
		name = f"{dense}d_{denseUnits}du_{self.interval}i_{self.timesteps}ts"
		self.model = keras.models.load_model(name)
		return self.model

	def predict(self,iden):
		updater= getData.Updater(interval,iden[0])
		data,_ = updater.processData(timesteps)
		data = getData.reshapeData(data)
		return self.model(data[-1])