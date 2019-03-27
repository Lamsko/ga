import numpy as np, random, operator, pandas as pd
import matplotlib.pyplot as plt


class City:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def distance(self, city):
		xDis = abs(self.x - city.x)
		yDis = abs(self.y - city.y)
		distance = np.sqrt((xDis ** 2) + (yDis ** 2))
		return distance

	def __repr__(self):
		return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
	def __init__(self, route):
		self.route = route
		self.distance = 0
		self.fitness = 0.0

	def routeDistance(self):
		if self.distance == 0:
			pathDistance = 0
			for i in range(0, len(self.route)):
				fromCity = self.route[i]
				toCity = None
				if i + 1 < len(self.route):
					toCity = self.route[i + 1]
				else:
					toCity = self.route[0]
				pathDistance += fromCity.distance(toCity)
			self.distance = pathDistance
		return self.distance

	def routeFitness(self):
		if self.fitness == 0:
			self.fitness = 1 / float(self.routeDistance())
		return self.fitness


def createRoute(cityList):
	route = random.sample(cityList, len(cityList))
	return route


def initialPopulation(popSize, cityList):
	population = []
	for i in range(0, popSize):
		population.append(createRoute(cityList))
	return population


def rankRoutes(population):
	fitnessResults = {}
	for i in range(0, len(population)):
		fitnessResults[i] = Fitness(population[i]).routeFitness()
	return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
	selectionResults = []
	df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
	df['cum_sum'] = df.Fitness.cumsum()
	df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
	for i in range(0, eliteSize):
		selectionResults.append(popRanked[i][0])
	for i in range(0, len(popRanked) - eliteSize):
		pick = 100 * random.random()
		for i in range(0, len(popRanked)):
			if pick <= df.iat[i,3]:
				selectionResults.append(popRanked[i][0])
				break
	return selectionResults

def matingPool(population, selectionResults):
	matingPool = []
	for i in range(0, len(selectionResults)):
		index = selectionResults[i]
		matingPool.append(population[index])
	return matingPool
