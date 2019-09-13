'''Based on the example of
https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
'''


# Wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
import csv
import sys





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





def main():
	result, cols = readCsv(sys.argv[1])
	#print(result, '\n\n')
	#print(cols)
	#print(result[cols['Cuda Value']], '\n\n')
	#print(result[cols['Normal-1 Value']])

	#Test if Values are significantly different
	print('---- Test for Values-----')
	data1 = list(map(float, result[cols['Cuda Value']]))
	data2 = list(map(float, result[cols['Normal-1 Value']]))
	testValues(data1, data2)

	#Test if Times are significantly different
	print('\n\n---- Test for Times Cuda x 1 Thread-----')
	data1 = list(map(float, result[cols['Cuda Time']]))
	data2 = list(map(float, result[cols['Normal-1 Time']]))
	testValues(data1, data2)


	#Test if Times are significantly different
	print('\n\n---- Test for Times Cuda x 4 Thread-----')
	data1 = list(map(float, result[cols['Cuda Time']]))
	data2 = list(map(float, result[cols['Normal-4 Time']]))
	testValues(data1, data2)


	#Test if Times are significantly different
	print('\n\n---- Test for Times Cuda x 8 Thread-----')
	data1 = list(map(float, result[cols['Cuda Time']]))
	data2 = list(map(float, result[cols['Normal-8 Time']]))
	testValues(data1, data2)

main()