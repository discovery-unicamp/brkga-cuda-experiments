import os
import sys
import time
import re
import datetime
import statistics as stat
import math

results_dir = ['executables-tsp/brkgaAPI-1/',  'executables-tsp/brkgaAPI-4/',
		  'executables-tsp/brkgaAPI-8/', 'executables-tsp/cuda-device-decode/', 
		  'executables-tsp/cuda-host-decode1/', 'executables-tsp/cuda-host-decode4/','executables-tsp/cuda-host-decode8/']
res_dir = 'results-tsp-cities1/'
results_dir = list(map(lambda x: x+res_dir, results_dir))
#print(results_dir)
algs_nick_names = ['brkga-tsp-1','brkga-tsp-4','brkga-tsp-8','cuda-device','cuda-host1','cuda-host4','cuda-host8']	


def main():
	d = str(datetime.datetime.now()).replace(' ', '-')
	if(not os.path.exists('results')):
		os.system('mkdir results')
	fout = open('results/results-TSP-'+d+'.csv', 'w')
	results_files = os.listdir(results_dir[0])
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
		for dirr in results_dir:
			times = []
			values = []
			for inst in inst_group:
				#print('open file',dirr+inst)
				try:
					f = open(dirr+inst, 'r')
					s = f.read()
					f.close()
				except Exception:
					s = ''
				value = exp_value.findall(s)
				time = exp_time.findall(s)
				if value == []:
					value = math.inf
				if time == []:
					time = math.inf
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

