import os
import sys
import time
import re
import datetime
import statistics as stat
import math

def getInstanceSize(arqName):
	exp = re.compile(r'\d\d+')
	return int(exp.search(arqName).group())

original_dir = 'instances/tsplib-vlsiAll/'
copy_dir = 'instances/tsplib-vlsi/'

if(not os.path.exists('instances/tsplib-vlsi/')):
	os.system('mkdir instances')
	os.system('mkdir instances/tsplib-vlsi')

results_files = os.listdir(original_dir)
total=0
for file in results_files:
	if getInstanceSize(file)>=3000 and getInstanceSize(file)<=20000:
		os.system('cp '+original_dir+file+' '+copy_dir+file)
		total += 1
print('Copied',total,'instances!')

