import urllib.request

print('Beginning file download with urllib2...')

url = 'http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/'

files = ['scp'+str(i)+'.txt' for i in range(41,66)]

for f in files:
	print('Downloading file:',f)
	try:
		urllib.request.urlretrieve(url+f, 'instances/instances-scp/'+f)
	except urllib.error.HTTPError:
		print('Could not download file:', f)