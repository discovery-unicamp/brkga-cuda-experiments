import os
import sys
import time
import re
import datetime

results_dir = ['executables-tsp/brkgaAPI-1/',  'executables-tsp/brkgaAPI-4/',
		  'executables-tsp/brkgaAPI-8/', 'executables-tsp/cuda-device-decode/', 
		  'executables-tsp/cuda-host-decode1/', 'executables-tsp/cuda-host-decode4/','executables-tsp/cuda-host-decode8/']
results_dir = list(map(lambda x: x+'results/', results_dir))
print(results_dir)
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

	for inst in results_files:
		fout.write(inst+',\t ')
		for dirr in results_dir:
			#print('open file',dirr+inst)
			try:
				f = open(dirr+inst, 'r')
				s = f.read()
				f.close()
			except Exception:
				s = ''
			value = exp_value.findall(s)
			time = exp_time.findall(s)
			if value != [] and time != []:
				fout.write(str(value[0])+', '+str(time[0])+', ')
			else:
				fout.write('None,\t None,\t ')
		fout.write('\n')
	fout.close()


def getInstanceSize(arqName):
	exp = re.compile(r'\d+')
	return int(exp.search(arqName).group())


main()

