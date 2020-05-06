To build the executable run "make".
It will be created a folder build/ and inside it the executable cuda-tsp.
To test the program you can execute ./build/cuda-tsp tsplib-cities/name-of-instance-file

The file ./config.txt has runtime configuration parameters for cuda-tsp. By default it
was set decode type to be done in the GPU (DEVICE_DECODE_CHROMOSOME_SORTED,decode_type 3). You can
test other options such as "decode_type 1" to use the host CPU to decode chromosomes.