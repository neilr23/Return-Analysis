# Neil Rayala
# 6/17/2020

import urllib.request
import json
import random
import csv
import dateutil.parser
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Server API URLs
QUERY = "http://localhost:8080/query?id={}"

# 500 server request
N = 500

def getDataPoint(quote):
	""" Produce all of the needed values to generate a datapoint """
	stock = quote['stock']
	bid_price = float(quote['top_bid']['price'])
	ask_price = float(quote['top_ask']['price'])
	price = (bid_price + ask_price)/2 # stock price is average of bid and ask price
	timestamp = quote['timestamp']
	return [timestamp, stock, price]

def returns_corr(stock_1, stock_2):
	""" Quantify the correlation between ABC's returns and DEF's returns """
	ptchange_1 = pd.Series([el[1] for el in stock_1]).pct_change()
	ptchange_2 = pd.Series([el[1] for el in stock_2]).pct_change()
	df = pd.concat([ptchange_1, ptchange_2], axis=1)
	X = df[0].values[:, np.newaxis].tolist()
	y = df[1].values.tolist()
	X.pop(0)
	y.pop(0)

	reg = LinearRegression().fit(X, y)
	corr_s = df.corr(method='spearman')[1][0] #Spearman Coefficient
	corr_p = df.corr(method='pearson')[1][0] #Pearson Coefficient

	plt.scatter(X, y)
	plt.xlabel('ABC Returns')
	plt.ylabel('DEF Returns')
	plt.plot(X, reg.predict(X), color='r')
	plt.legend(['Slope: ' + str(round(reg.coef_[0], 5))])
	plt.savefig('returns_plot.png')
	plt.show()

	return corr_s, corr_p

def getData():
	""" Retreive the necessary data for ABC and DEF """
	stock_1 = []
	stock_2 = []

	with open('data.csv', 'r') as f:
		for r in csv.reader(f, delimiter=','):
			timestamp = r[0]
			stock = r[1]
			price = r[2]
			if stock == 'ABC':
				stock_1.append([dateutil.parser.parse(timestamp), float(price)])
			elif stock == 'DEF':
				stock_2.append([dateutil.parser.parse(timestamp), float(price)])

	stock_1 = sorted(stock_1) #sort based on oldest
	stock_2 = sorted(stock_2)
	return stock_1, stock_2

def writeData():
	""" Generate the necessary data for ABC and DEF"""
	with open('data.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['timestamp', 'stock', 'bid_price', 'ask_price'])

		# Query the price once every N seconds.
		for _ in iter(range(N)):
			quotes = json.loads(urllib.request.urlopen(QUERY.format(random.random())).read())

			for quote in quotes:
				writer.writerow(getDataPoint(quote))

# Main
if __name__ == '__main__':
	if not os.path.isfile('data.csv'):
		print("No data found, generating...")
		writeData()

	stock_1, stock_2 = getData()
	corr_s, corr_p = returns_corr(stock_1, stock_2)
	print('Pearson Correlation between ABC returns and DEF returns:', corr_p)
	print('Spearman Correlation between ABC returns and DEF returns:', corr_s)