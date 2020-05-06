The source codes are in the folder src:
- src/Decoder.cu contains the definitions of 3 decoder functions, of which at least one must be implemented, depending on where the decode is going to be executed, on the host, on the device  or on the device with chromosomes sorted.
- src/BRKGA.cu contains the main implementation of the BRKGA algorithm, it should not be modified.
- other than the decoder, a user must only implement its main function program and other related functions related to its specific problem.