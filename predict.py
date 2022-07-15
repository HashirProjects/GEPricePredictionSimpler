import getData
from tensorflow import keras

class Predictor():

	def __init__(self,interval,timesteps):
		self.timesteps = timesteps
		self.interval = interval

	def getmodel(self, dense, denseUnits):
		name = f"{dense}d_{denseUnits}du_{self.interval}i_{self.timesteps}ts"
		self.model = keras.models.load_model(name)
		return self.model

	def predict(self,iden):#best to make an array of last data points and pass them into the model at the same time
		updater= getData.Updater(self.interval,iden)
		self.data,_ = updater.processData(self.timesteps)
		return self.model(self.data)[-1]

	def plotfinal(self):
		import matplotlib.pyplot as plt
		plt.plot(self.data[-1])
		plt.show()

predictor = Predictor('24h',30)
predictor.getmodel(2,32)
print(predictor.predict(4151))
predictor.plotfinal()
