import os
import sys
import time
import re
import datetime
import statistics as stat
import math

SCP = True
TSP = False

if SCP:
	executables_dir = ['executables-scp/brkgaAPI-1/',  'executables-scp/brkgaAPI-4/', 'executables-scp/brkgaAPI-8/',
		  'executables-scp/cuda-host-decode1/', 'executables-scp/cuda-host-decode4/','executables-scp/cuda-host-decode8/']
	running_results_dir = 'results-scp-testscp/'
	algs_nick_names = ['brkga-scp-1','brkga-scp-4','brkga-scp-8','cuda-host1','cuda-host4','cuda-host8']
	csv_dir = 'results/scp/'
elif TSP:
	executables_dir = ['executables-tsp/brkgaAPI-1/',  'executables-tsp/brkgaAPI-4/','executables-tsp/brkgaAPI-8/',
		 'executables-tsp/cuda-device-decode/', 
		 'executables-tsp/cuda-host-decode1/', 'executables-tsp/cuda-host-decode4/','executables-tsp/cuda-host-decode8/']
	running_results_dir = 'results-tsp-testtsp/'
	algs_nick_names = ['brkga-tsp-1','brkga-tsp-4','brkga-tsp-8','cuda-device','cuda-host1','cuda-host4','cuda-host8']	
	csv_dir = 'results/tsp/'

runnings_dir = list(map(lambda x: x+running_results_dir, executables_dir))

if(not os.path.exists('results')):
	os.system('mkdir results')
	os.system('mkdir results/tsp')
	os.system('mkdir results/scp')


def main():
	d = str(datetime.datetime.now()).replace(' ', '-')
	fout = open(csv_dir+'results'+d+'.csv', 'w')
	results_files = os.listdir(runnings_dir[0])
	results_files.sort(key=getInstanceSize)
	exp_value = re.compile(r'Value of best solution: *(\d+.\d+)')
	exp_time = re.compile(r'Time: *(\d+.?\d*)')

	fout.write('Instance Name')
	for alg in algs_nick_names:
		fout.write(',\t '+alg+' Value,\t '+alg+' Time')
	fout.write('\n')

	results_files = merge_same_instances(results_files)
	for inst_group in results_files:
		fout.write(inst_group[0][:-1]+',\t ')
		for dirr in runnings_dir:
			times = []
			values = []
			for inst in inst_group:
				print('processing file',dirr+inst)
				try:
					f = open(dirr+inst, 'r')
					s = f.read()
					f.close()
				except Exception:
					s = ''
				value = exp_value.findall(s)
				time = exp_time.findall(s)
				if value == []:
					value = [math.inf]
				if time == []:
					time = [math.inf]
				times.append(float(time[0]))
				values.append(float(value[0]))
			v_m = "{0:.2f}".format(stat.mean(values))
			v_d = "{0:.2f}".format(stat.stdev(values))
			t_m = "{0:.2f}".format(stat.mean(times))
			t_d = "{0:.2f}".format(stat.stdev(times))
			fout.write(v_m+' ('+ v_d +')' +', ' +t_m+ ' ('+ t_d +')'+', ')
		fout.write('\n')
	fout.close()


def merge_same_instances(results_files):
	l = []
	ini = results_files[0][:-1]
	aux = [results_files[0]]
	for i in range(1,len(results_files)):
		if results_files[i][:-1] == ini:
			aux.append(results_files[i])
		else:
			l.append(aux)
			aux = [results_files[i]]
			ini = results_files[i][:-1]
	l.append(aux)
	return l


def getInstanceSize(arqName):
	exp = re.compile(r'\d\d+')
	return int(exp.search(arqName).group())


main()

