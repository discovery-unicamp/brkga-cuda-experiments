To compile do:
 mkdir build
 cd build
 cmake .. -DCMAKE_BUILD_TYPE=release
 make

 To execute the TSP examples:
  pwd
  ~/build
  ./brkga-tsp -p ../config-tsp.txt -i ../tsplib-cities/lu980.tsp
    or
	./brkga-tsp -p ../config-tsp.txt -i ../tsplib-cities/lu980.tsp -c #to use coalesced next population
	  or
	./brkga-tsp -p ../config-tsp.txt -i ../tsplib-cities/lu980.tsp -l 1 #to use pipeline with 1 population decoded on GPU


To execute the SCP examples:
	In the build directory execute
	./brkga-scp -p ../config-scp.txt -i ../scplib/scp41.txt
  or
	./brkga-scp -p ../config-scp.txt -i ../scplib/scp41.txt -c #to use coalesced next population
	or
	./brkga-scp -p ../config-scp.txt -i ../scplib/scp41.txt -l 0 #to use pipeline with all populations decoded on host
