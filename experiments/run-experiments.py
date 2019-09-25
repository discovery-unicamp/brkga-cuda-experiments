import os
import sys
import time

NUMBER_RUNS = 3 #number of times to solve each instance

SCP = False
TSP = True

if SCP:
	algs_dir = ['executables-scp/brkgaAPI-1/',  'executables-scp/brkgaAPI-4/','executables-scp/brkgaAPI-8/',   
		    'executables-scp/cuda-host-decode1/','executables-scp/cuda-host-decode4/','executables-scp/cuda-host-decode8/']
	algs_names = ['brkga-scp','brkga-scp','brkga-scp','cuda-scp','cuda-scp','cuda-scp']
	algs_nick_names = ['brkga-scp-1','brkga-scp-4','brkga-scp-8','cuda-host1','cuda-host4','cuda-host8']	
	inst_dir = 'instances/testscp/'
	results_dir = 'results-scp-testscp/'
	brkgaAPIparam  = ' 1234 0 100 ' #random-seed 0-is max generations 100-is the number of max generations
elif TSP:
	algs_dir = ['executables-tsp/brkgaAPI-1/',  'executables-tsp/brkgaAPI-4/', 'executables-tsp/brkgaAPI-8/',
	     'executables-tsp/cuda-device-decode/',
		    'executables-tsp/cuda-host-decode1/','executables-tsp/cuda-host-decode4/','executables-tsp/cuda-host-decode8/']
	algs_names = ['brkga-tsp','brkga-tsp','brkga-tsp','cuda-tsp','cuda-tsp','cuda-tsp','cuda-tsp']
	algs_nick_names = ['brkga-tsp-1','brkga-tsp-4','brkga-tsp-8','cuda-device','cuda-host1','cuda-host4','cuda-host8']	
	inst_dir = 'instances/tsplib-cities/'
	results_dir = 'results-tsplib-cities/'
	brkgaAPIparam  = ''

current_dir = os.getcwd()


def main():
	#dire = sys.argv[1]
	#os.chdir(cmd)
	instances_files = os.listdir(inst_dir)

	for i in range(len(algs_dir)):
		print('Processing Alg',algs_nick_names[i])
		os.chdir(algs_dir[i])
		if os.path.exists(results_dir):
			cmd = 'rm '+results_dir+'*'
			os.system(cmd)
		else:
			os.system('mkdir '+results_dir)
		for j in range(NUMBER_RUNS):
			for instance in instances_files:
				if algs_names[i] == 'brkga-scp':
					cmd = './'+algs_names[i]+brkgaAPIparam+' '+current_dir+'/'+inst_dir+instance +' > '+results_dir+instance+'.result'+str(j)
				else:
					cmd = './'+algs_names[i]+' '+current_dir+'/'+inst_dir+instance +' > '+results_dir+instance+'.result'+str(j)					
				start = time.time()
				os.system(cmd)
				end = time.time()
				print('Finished ',instance)
				f = open(results_dir+instance+'.result'+str(j), 'a')
				f.write('Total Time: '+str(end-start))
				f.close()
		os.chdir(current_dir)

main()