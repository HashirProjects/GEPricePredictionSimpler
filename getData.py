import requests
import numpy as np
import pickle

#import sys
#np.set_printoptions(threshold=sys.maxsize)
#Above can be uncommented to fully display np arrays

class Updater():
	"""fetches and processes the market data for 1 item"""
	headers = {
		'User-Agent': 'GE price forcasting project. email: hashirrana2001@gmail.com',
	}

	def __init__(self,timestep,itemID):

		self.URL = f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep={timestep}&id={itemID}"
		r = requests.get(self.URL, headers=self.headers)
		self.unprocessed = r.json()


	def getUnprocessed(self):
		return self.unprocessed

	def processData(self,timesteps):#perhaps should save only avgprices? simplify: make labels binary and pass in only avg prices

		self.values = []
		self.labels = []
		#self.AvgPrices = []

		for i in range(len(self.unprocessed["data"])):
			try:

				if self.unprocessed["data"][i]['avgHighPrice'] == None or self.unprocessed["data"][i]['avgLowPrice'] == None:
					raise(TypeError) 

				AvgPrice = (self.unprocessed["data"][i]['avgHighPrice']*self.unprocessed["data"][i]['highPriceVolume']+self.unprocessed["data"][i]['avgLowPrice']*self.unprocessed["data"][i]['lowPriceVolume'])/(self.unprocessed["data"][i]['highPriceVolume']+self.unprocessed["data"][i]['lowPriceVolume'])

				self.values.append(AvgPrice)
			
			except TypeError:
				print(f"data at {self.unprocessed['data'][i]['timestamp']} was unsuitable")


		self.values = np.array(self.values)

		self.valuesAggregated = []

		for j in range(len(self.values)-timesteps):

			self.valuesAggregated.append([])
			for i in range(timesteps):
				self.valuesAggregated[j].append(self.values[j+i])

		self.values = np.array(self.valuesAggregated)


		#prevlen = self.values.shape
		#print(prevlen)
		#samples = prevlen[0]//timesteps
		#self.values = np.resize(self.values,(samples,timesteps,prevlen[1]))
		#self.AvgPrices = np.resize(self.values,(samples,timesteps,1))

		#print(self.AvgPrices)

		for i in range(len(self.values)-1):

			# currentvalue = self.values[i][-1]
			# nextvalue = self.values[i+1][0]
			# currentavg = ((currentvalue[0]*currentvalue[2])+(currentvalue[1]*currentvalue[3]))/(currentvalue[2]+currentvalue[3])
			# nextavg = ((nextvalue[0]*nextvalue[2])+(nextvalue[1]*nextvalue[3]))/(nextvalue[2]+nextvalue[3])

			currentavg = self.values[i][-1]
			nextavg = self.values[i+1][-1]

			if nextavg > currentavg:
				self.labels.append(1)
			else:
				self.labels.append(0)

		self.values = self.values / np.amax(self.values)#normalise data to increase training speed

		self.values = self.values[:-1] # the last value doesnt have a lalbel since the next avg cant be calculated

		return self.values, np.array(self.labels)

def randomiseInUnison(a,b): 
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

def pickleData(data,filename):
	with open(filename,"wb") as file:
		pickle.dump(data,file)

def reshapeData(data):
	prevsize = data.shape
	return np.reshape(data,[prevsize[0],-1])

if __name__ == "__main__":

	updater= Updater("5m",449)
	a,b = updater.processData(30)

	for i in [4151]: #include all the different item ids you want to save
		updater= Updater("5m",i)
		tempa,tempb = updater.processData(30)
		a  = np.concatenate((a,tempa))
		b  = np.concatenate((b,tempb))

	print(a,b)
	a,b = randomiseInUnison(a,b)

	#pickleData(a,"Data.txt")
	#pickleData(b,"Labels.txt")


	#c = np.array(list(zip(a,b)))
	#print(c)

	#with open("db.txt","wb") as file:
		#pickle.dump(c,file)


