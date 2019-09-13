import os
import sys
import time


algs_dir = ['executables-tsp/brkgaAPI-1/',  'executables-tsp/brkgaAPI-4/',
		  'executables-tsp/brkgaAPI-8/',   'executables-tsp/cuda-device-decode/',
		    'executables-tsp/cuda-host-decode1/','executables-tsp/cuda-host-decode4/','executables-tsp/cuda-host-decode8/']
algs_names = ['brkga-tsp','brkga-tsp','brkga-tsp','cuda-tsp','cuda-tsp','cuda-tsp','cuda-tsp']
algs_nick_names = ['brkga-tsp-1','brkga-tsp-4','brkga-tsp-8','cuda-device','cuda-host1','cuda-host4','cuda-host8']	
instances_dir = 'instances/tsplib-cities-test/'
current_dir = os.getcwd()

def main():
	#dire = sys.argv[1]
	#os.chdir(cmd)
	cities_files = os.listdir(instances_dir)

	for i in range(len(algs_dir)):
		print('Processing Alg',algs_nick_names[i])
		os.chdir(algs_dir[i])
		if(not os.path.exists('results')):
			os.system('mkdir results')
		for city in cities_files:
			cmd = './'+algs_names[i]+ ' '+current_dir+'/'+instances_dir+city +' > results/'+city+'.result'
			start = time.time()
			os.system(cmd)
			end = time.time()
			print('Finished ',city)
			f = open('results/'+city+'.result', 'a')
			f.write('Total Time: '+str(end-start))
			f.close()
		os.chdir(current_dir)

main()