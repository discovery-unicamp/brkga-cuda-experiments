import os
import sys
import time
import re
import datetime

def main():
	d = str(datetime.datetime.now()).replace(' ', '-')
	fout = open('results-VLSI-'+d+'.csv', 'w')
	results_files = os.listdir('cuda/resultsVLSI/')
	results_files.sort(key=getInstanceSize)
	exp_value = re.compile(r'Value of best solution: *(\d+.\d+)')
	exp_time = re.compile(r'Time: *(\d+.?\d*)')
	fout.write(' , Cuda Value, Cuda Time, Normal-1 Value, Normal-1 Time, Normal-4 Value, Normal-4 Time, Normal-8 Value, Normal-8 Time\n')
	for r in results_files:
		fout.write(r+', ')
		try:
			f = open('cuda/resultsVLSI/'+r, 'r')
			s = f.read()
			f.close()
		except Exception:
			s = ''
		value = exp_value.findall(s)
		time = exp_time.findall(s)
		if value != [] and time != []:
			fout.write(str(value[0])+', '+str(time[0])+', ')
		else:
			fout.write('None, None, ')


		try:
			f = open('normal-1/resultsVLSI/'+r, 'r')
			s = f.read()
			f.close()
		except Exception:
			s = ''
		value = exp_value.findall(s)
		time = exp_time.findall(s)
		if value != [] and time != []:
			fout.write(str(value[0])+', '+str(time[0])+', ')
		else:
			fout.write('None, None, ')


		try:
			f = open('normal-4/resultsVLSI/'+r, 'r')
			s = f.read()
			f.close()
		except Exception:
			s = ''
		value = exp_value.findall(s)
		time = exp_time.findall(s)
		if value != [] and time != []:
			fout.write(str(value[0])+', '+str(time[0])+', ')
		else:
			fout.write('None, None, ')

		try:
			f = open('normal-8/resultsVLSI/'+r, 'r')
			s = f.read()
			f.close()
		except Exception:
			s = ''
		value = exp_value.findall(s)
		time = exp_time.findall(s)
		if value != [] and time != []:
			fout.write(str(value[0])+', '+str(time[0]))
		else:
			fout.write('None, None ')

		fout.write('\n')
	fout.close()



def getInstanceSize(arqName):
	exp = re.compile(r'\d+')
	return int(exp.search(arqName).group())


main()

