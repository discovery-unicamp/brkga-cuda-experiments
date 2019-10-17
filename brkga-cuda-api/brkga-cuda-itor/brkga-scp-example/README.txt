To build the executable run "make".
It will be created a folder build/ and inside it the executable cuda-scp.
To test the program you can execute 

	./build/cuda-scp ./instances/orlib/scp41.txt

Additional stopping rules are: 1, to stop after a target objective value is foun
d, and 2, to stop
after a given number of generations has passed without improvement in the best o
bjective value.

To set up the BRKGA parameters, edit file algorithm.conf con contain exactly the
 following data in
the first 7 lines:

0.20 (percentage of the population labeled elite)
0.15 (percentage of the population introduced at random at each generation, the 
mutants)
0.80 (when applying cross-over, probability that an allele will be copied from t
he elite parent)
2 (number of independent populations)
2 (number of threads employed to decode chromosomes in parallel)
10 (frequency at which elite individuals are exchanged between the populations)
2 (number of elite individuals to exchange)

The size of the population is determined by the size of the problem n (number of
 rows): 10 * n.



The file ./config.txt has runtime configuration parameters for cuda-tsp. By default it
was set decode type to be done in the GPU (DEVICE_DECODE_CHROMOSOME_SORTED,decode_type 3). You can
test other options such as "decode_type 1" to use the host CPU to decode chromosomes.