import csv

filename = 'results/pc-resultsTSPcicites.csv'

class DictReaderStrip(csv.DictReader):
    @property                                    
    def fieldnames(self):
        if self._fieldnames is None:
            # Initialize self._fieldnames
            # Note: DictReader is an old-style class, so can't use super()
            csv.DictReader.fieldnames.fget(self)
            if self._fieldnames is not None:
                self._fieldnames = [name.strip() for name in self._fieldnames]
        return self._fieldnames


with open(filename, newline='') as csvfile:
	reader = DictReaderStrip(csvfile)
	for row in reader:
		t1 = row['brkga-tsp-8 Time']
		t2 = row['cuda-host8 Time']
		t1 = float(t1[: t1.find('(')])
		t2 = float(t2[: t2.find('(')])
		print('brkga:',t1,'cuda:',t2,' %faster:',(t1-t2)/t2)
