import os
import sys
import time
import re

def main():
	#dire = sys.argv[1]
	#os.chdir(cmd)
	vlsi_files = os.listdir('tsplib-vlsi')
	vlsi_files2 = []
	for i in vlsi_files:
		if instanceSize(i)>25000:
			print('Not considering instance',i,'with more than 25000 nodes (overflow GPU memory)')
		else:
			vlsi_files2.append(i)
	vlsi_files = vlsi_files2

	#execute cuda
	os.chdir('cuda')
	for i in vlsi_files:
		print('Cuda processing',i)
		cmd = './app ../tsplib-vlsi/'+i+' > resultsVLSI/'+i+'.result'
		start = time.time()
		os.system(cmd)
		end = time.time()
		f = open('resultsVLSI/'+i+'.result', 'a')
		f.write('Total Time: '+str(end-start))


	os.chdir('..')
	os.chdir('normal-1')
	for i in vlsi_files:
		print('Normal-1 processing',i)
		cmd = './brkga-tsp ../tsplib-vlsi/'+i+' > resultsVLSI/'+i+'.result'
		start = time.time()
		os.system(cmd)
		end = time.time()
		f = open('resultsVLSI/'+i+'.result', 'a')
		f.write('Total Time: '+str(end-start))

	os.chdir('..')
	os.chdir('normal-4')
	for i in vlsi_files:
		print('Normal-4 processing',i)
		cmd = './brkga-tsp ../tsplib-vlsi/'+i+' > resultsVLSI/'+i+'.result'
		start = time.time()
		os.system(cmd)
		end = time.time()
		f = open('resultsVLSI/'+i+'.result', 'a')
		f.write('Total Time: '+str(end-start))

	os.chdir('..')
	os.chdir('normal-8')
	for i in vlsi_files:
		print('Normal-8 processing',i)
		cmd = './brkga-tsp ../tsplib-vlsi/'+i+' > resultsVLSI/'+i+'.result'
		start = time.time()
		os.system(cmd)
		end = time.time()
		f = open('resultsVLSI/'+i+'.result', 'a')
		f.write('Total Time: '+str(end-start))



def instanceSize(s):
	exp = re.compile(r'(\d+)')
	return int(exp.findall(s)[0])

main()