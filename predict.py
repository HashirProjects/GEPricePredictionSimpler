import getData
from tensorflow import keras
import matplotlib.pyplot as plt

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
		self.data = updater.calcAvgValues(self.timesteps)
		self.results = self.model(self.data)
		return self.results

	def plotall(self):
		for i in range(len(self.data)):
			print(self.results[i])			
			plt.plot(self.data[i])
			plt.show()

	def plotfinal(self):
		print(self.results[-1])			
		plt.plot(self.data[-1])
		plt.show()

	def plotdata(self):
		plt.plot(self.data)
		plt.show()

if __name__ == "__main__":
	predictor = Predictor('24h',7)
	predictor.getmodel(3,32)
	predictor.predict(4151)
	predictor.plotdata()
	predictor.plotfinal()
