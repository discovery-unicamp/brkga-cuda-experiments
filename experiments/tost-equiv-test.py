'''Based on the example of
https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
'''

#tost test based on https://www.biochemia-medica.com/assets/images/upload/xml_tif/bm-27-5.pdf

# Wilcoxon signed-rank test
import numpy as np
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
from statsmodels.stats.weightstats import ttost_ind
import csv
import sys
from random import randint as rand


def exampleTest():
	# seed the random number generator
	seed(1)
	# generate two independent samples
	data1 = 5 * randn(100) + 50
	data2 = 5 * randn(100) + 51
	print(data1)

	# compare samples
	stat, p = wilcoxon(data1, data2)
	print('Statistics=%.3f, p=%.3f' % (stat, p))
	# interpret
	alpha = 0.05
	if p > alpha:
		print('Same distribution (fail to reject H0)')
	else:
		print('Different distribution (reject H0)')

def testEquivalence(data1, data2, low, up,transform=None):
	p = ttost_ind(data1, data2,low,up,transform=transform)
	print(p)
	print('p-value:',p[0])
	if(p[0]<.05):
		print('Same distribution')
	else:
		print('Different distribution')

	#print(ttost_ind(data1, data2,10,10,usevar='unequal'))

#wilcoxon test
def testValues(data1, data2):
	stat, p = wilcoxon(data1, data2)
	print('Statistics=%.3f, p=%.3f' % (stat, p))
	alpha = 0.05
	if p > alpha:
		print('Same distribution (fail to reject H0)')
	else:
		print('Different distribution (reject H0)')


def readCsv(fileName):
	with open(fileName) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		cols = {}
		result = []
		for row in csv_reader:
			#print(row)
			if line_count == 0:
				colIndex = 0
				for column in row:
					cols[column.strip()] = colIndex
					colIndex += 1
					result.append([])
				line_count += 1
			else:
				if not containsNone(row):
					#print(row)
					for i in range(len(row)):
						result[i].append(row[i].strip())
				line_count += 1
	print('Processed',line_count,'lines!')
	return result, cols

def containsNone(row):
	for r in row:
		if r.strip()=='None':
			return True
	return False



def cohensd_effect_size(experimental_group, control_group):
	mean_e = np.mean(experimental_group)
	mean_c = np.mean(control_group)
	effect_size = (mean_e - mean_c)/np.std(control_group)
	if effect_size < 0 :


def main():
	result, cols = readCsv(sys.argv[1])
	#print(result, '\n\n')
	#print(cols)
	#print(result[cols['brkga-tsp-1 Value']], '\n\n')
	#print(result[cols['cuda-device Value']])

	#Test if Values are significantly different
	print('---- Test for Values-----')
	data1 = list(map(float, result[cols['brkga-tsp-1 Value']]))
	data2 = list(map(float, result[cols['cuda-device Value']]))
	testValues(data1, data2)

	print('---- Test for Equivalence-----')
	data1 = list(map(float, result[cols['brkga-tsp-1 Value']]))
	data2 = list(map(float, result[cols['cuda-device Value']]))
	print(data1)
	print(data2)
	testEquivalence(data1, data2,0.95,1.05,transform=np.log)


	# print('---- Test for Equivalence-----')
	# data1 = []
	# data2 = []
	# for i in range(100):
	# 	data1.append(rand(0,100))
	# 	#data2.append(rand(0,100))
	# data2 = data1.copy()
	# print(data1)
	# print(data2)
	# testEquivalence(data1, data2)



main()