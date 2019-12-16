import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

figs_dir = 'figs'

def main():
	if(len(sys.argv)!=2):
		print('Usage \'python generate-figures.py path_file.csv')
		return
	df = pd.read_csv(sys.argv[1],sep=',')
	df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
	print(df.columns.values)

	cudaTimes = df['cuda-device_time'].values
	cpu1Times = df['brkga-tsp-1_time'].values
	cpu4Times = df['brkga-tsp-4_time'].values
	cpu8Times = df['brkga-tsp-8_time'].values

	#plt.style.use('seaborn-whitegrid')
	print('Number of instances:' , len(cudaTimes))
	x = np.linspace(0,len(cudaTimes)-1,len(cudaTimes))
	fig = plt.figure()
	ax = plt.axes()
	ax.plot(x,cudaTimes, linestyle='solid', label='cuda')
	ax.plot(x,cpu1Times, linestyle='dashed', label='cpu-1')
	ax.plot(x,cpu4Times, linestyle='dashdot', label='cpu-4')
	ax.plot(x,cpu8Times, linestyle='dotted', label='cpu-8')
	plt.legend()
	ax.set_yscale('log')
	plt.xlim(0,len(cudaTimes))
	plt.xlabel('instances')
	plt.ylabel('time (s)')

	xlabels = []
	xticks = []
	for i in range(0,len(cudaTimes),5):
		xticks.append(i)
		xlabels.append('i'+str(i))
	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabels, rotation=90)

	fig.savefig(figs_dir+'/fig-execution-times.png')

main()














def main_old():
	#the argument is the file containing the results in a csv format
	result, cols = readCsv(sys.argv[1])
	#print(result, '\n\n')
	#print(cols)
	#print(result[cols['Cuda Value']], '\n\n')
	#print(result[cols['Normal-1 Value']])
	#plt.style.use('seaborn-whitegrid')
	print('Number of instances:' , len(cudaTimes))
	x = np.linspace(0,len(cudaTimes)-1,len(cudaTimes))
	fig = plt.figure()
	ax = plt.axes()
	ax.plot(x,cudaTimes, linestyle='solid', label='cuda')
	ax.plot(x,cpu1Times, linestyle='dashed', label='cpu-1')
	ax.plot(x,cpu4Times, linestyle='dashdot', label='cpu-4')
	ax.plot(x,cpu8Times, linestyle='dotted', label='cpu-8')
	plt.legend()
	ax.set_yscale('log')
	plt.xlim(0,len(cudaTimes))
	plt.xlabel('instances')
	plt.ylabel('time (s)')

	xlabels = []
	xticks = []
	for i in range(0,len(cudaTimes),5):
		xticks.append(i)
		xlabels.append('i'+str(i))
	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabels, rotation=90)

	fig.savefig('fig-execution-times.png')

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
