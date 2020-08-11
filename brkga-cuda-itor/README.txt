To compile do:
 mkdir build
 cd build
 cmake .. -DCMAKE_BUILD_TYPE=release
 make

 To execute do:
  pwd
  ~/build
  cd ..
  ./build/brkga-tsp -p config.txt -i tsplib-cities/lu980.tsp 
    or
  ./build/brkga-tsp -p config.txt -i tsplib-cities/lu980.tsp -c #to use coalesced next population