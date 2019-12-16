#tost test based on https://www.biochemia-medica.com/assets/images/upload/xml_tif/bm-27-5.pdf
#and NPAR test based on https://yorkspace.library.yorku.ca/xmlui/bitstream/handle/10315/34591/Mara%20%26%20Cribbie.pdf?sequence=1&isAllowed=y


# Wilcoxon signed-rank test
import numpy as np
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
from scipy.stats import t
from statsmodels.stats.weightstats import ttost_ind
import csv
import sys
from random import randint as rand
import pandas as pd
import matplotlib.pyplot as plt

TSP = False

if TSP:
	brkga1v = 'brkga-tsp-1_value'
	brkga1t = 'brkga-tsp-1_time'
	brkga4v = 'brkga-tsp-4_value'
	brkga4t = 'brkga-tsp-4_time'
	brkga8v = 'brkga-tsp-8_value'
	brkga8t = 'brkga-tsp-8_time'
	cudav = 'cuda-device_value'
	cudat = 'cuda-device_time'
else:
	brkga1v = 'brkga-1_value'
	brkga1t = 'brkga-1_time'
	brkga4v = 'brkga-4_value'
	brkga4t = 'brkga-4_time'
	brkga8v = 'brkga-8_value'
	brkga8t = 'brkga-8_time'
	cudav = 'cuda-host8_value'
	cudat = 'cuda-host8_time'

#wilcoxon test
def wilcoxon_test(data1, data2):
	stat, p = wilcoxon(data1, data2)
	print('Statistics=%.3f, p=%.3f' % (stat, p))
	alpha = 0.05
	if p > alpha:
		print('Same distribution (fail to reject H0 (distributions are the same))')
	else:
		print('Different distribution (reject H0 (distributions are the same))')


def tost_test(control, alternative, alpha=0.05, delta=0.05):
	'''
		Performs the tost test as in https://www.biochemia-medica.com/assets/images/upload/xml_tif/bm-27-5.pdf
		'The logic of equivalence testing and its use in laboratory medicine, by Cristiano Ialongo'
		alpha is the significance level in the t-test (default=5%)
		delta is the percentage around the control group values that is considered to be equivalent (default=5%)
		Important: as t-students test assume distributions are normal
	'''
	print(alternative)
	print(control)
	diff = np.array([a-c for a,c in zip(alternative,control)])
	d = np.mean(diff)
	print('Mean difference:',d)
	std =  np.std(diff)
	print('Std of differences:', std)
	deltae = delta*np.mean(control) #test if they are the same when the difference is in between +-5% of the mean
	print('Delta error:',deltae)
	tost_inf = (d - (-1*deltae))/(std * np.sqrt(1/len(diff)))
	tost_up = (deltae - d)/(std * np.sqrt(1/len(diff)))
	print('Tost_inf:',tost_inf,' Tost_up:',tost_up)
	degree_freedom = len(control)-1
	critival_value = t.ppf(1.0-alpha,degree_freedom)
	print('Critical Value:',critival_value)
	if abs(tost_inf) > critival_value and abs(tost_up) > critival_value:
		print('Distributions are equivalent (rejected H0 (differences are significative))')
	else:
		print('Cannot declare distributions are equivalent')

def npar_test(control, alternative, alpha=0.05, delta=0.05):
	'''
		Performs the npar test as in https://yorkspace.library.yorku.ca/xmlui/bitstream/handle/10315/34591/Mara%20%26%20Cribbie.pdf?sequence=1&isAllowed=y
		'Paired-Samples Tests of Equivalence, by CONSTANCE A. MARA AND ROBERT A. CRIBBIE'
		alpha is the significance level in the t-test (default=5%)
		delta is the percentage around the control group values that is considered to be equivalent (default=5%)
	'''
	pass

def main():
	if(len(sys.argv)!=2):
		print('Usage \'python generate-statistics.py path_file.csv')
		return
	df = pd.read_csv(sys.argv[1],sep=',')
	df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
	print(df.columns.values)
	#print(df['instance_name'].values)
	#print(df.iloc[:,1].values)


	#perform test to check is solutions values are equivalent
	x = df[brkga1v].values
	y = df[cudav].values
	print('\n\nResults of TOST test for equivalence')
	tost_test(x,y)
	#sanity check: test wilcoxon for different
	print('\n\nPerforming sanity chek with a wilcoxon test')
	wilcoxon_test(x,y)


	#perform test to check if computational times are significantly different
	x = df[brkga1t].values
	y = df[cudat].values
	#x = [i for i in range(20)]
	#y = [i+0.01 for i in range(10)] + [i-0.01 for i in range(10)]
	print('\n\nResults of Wilcoxon test for different distributions of brkga-tsp-1_time x cuda-device_time')
	wilcoxon_test(x,y)


	#perform test to check if computational times are significantly different
	x = df[brkga4t].values
	y = df[cudat].values
	#x = [i for i in range(20)]
	#y = [i+0.01 for i in range(10)] + [i-0.01 for i in range(10)]
	print('\n\nResults of Wilcoxon test for different distributions of brkga-tsp-4_time x cuda-device_time')
	wilcoxon_test(x,y)

	#perform test to check if computational times are significantly different
	x = df[brkga8t].values
	y = df[cudat].values
	#x = [i for i in range(20)]
	#y = [i+0.01 for i in range(10)] + [i-0.01 for i in range(10)]
	print('\n\nResults of Wilcoxon test for different distributions of brkga-tsp-8_time x cuda-device_time')
	wilcoxon_test(x,y)

main()



def testEquivalence(data1, data2, low, up,transform=None):
	p = ttost_ind(data1, data2,low,up,transform=transform)
	print(p)
	print('p-value:',p[0])
	if(p[0]<0.05):
		print('Same distribution (rejected H0 (distributions are different))')
	else:
		print('Different distribution (fail to rejected H0 (distributions are different))')

	#print(ttost_ind(data1, data2,10,10,usevar='unequal'))