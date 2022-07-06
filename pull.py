import pickle

def getDB(filename):
	with open(filename,"rb") as file:
		content = pickle.load(file)

	return content


if __name__ == "__main__":
	data = getDB("Data.txt")
	labels = getDB("Labels.txt")
	print(data)
