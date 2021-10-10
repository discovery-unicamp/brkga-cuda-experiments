/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 */

#ifndef CONFIGFILE_H
#define CONFIGFILE_H

#include <stdexcept>

#define POOL_SIZE 10 // size of the pool with the best solutions so far

#define HOST_DECODE                                                            \
  1 /// decoding is done on CPU (host), and user must implement a host_decode
    /// method in Decoder.
#define DEVICE_DECODE                                                          \
  2 /// decoding is done no GPU (device), and user must implement a
    /// device_decode method in Decoder.
#define DEVICE_DECODE_CHROMOSOME_SORTED                                        \
  3 /// decoding is done on GPU, and chromosomes are given sorted by genes values.
    /// Users should implement device_decode_chromosome_sorted.

/**
 * \brief ConfigFile contains all parameters to execute the algorithm. These
 * parameters are read from a config.txt file.
 */
class ConfigFile {
public:
  typedef std::runtime_error Error;

  explicit ConfigFile(const char *instanceFile);
  virtual ~ConfigFile() = default;

  unsigned p; /// size of population, example 256 individuals
  float pe;   /// proportion of elite population, example 0.1
  float pm;   /// proportion of mutant population, example 0.05
  float
      rhoe; /// probability that child gets an allele from elite parent, exe 0.7
  unsigned K;        /// number of different independent populations
  unsigned MAX_GENS; /// execute algorithm for MAX_GENS generations
  unsigned X_INTVL;  /// exchange best individuals at every X_INTVL generations
  unsigned X_NUMBER; /// exchange top X_NUMBER best individuals
  unsigned RESET_AFTER; /// restart strategy; reset all populations after this
                        /// number of iterations

  unsigned decode_type;  /// run decoder on GPU or Host, see decode_t enum
  unsigned decode_type2; /// when using pipelining this is the second decode
                         /// type to be performed on GPU
  unsigned OMP_THREADS;  /// number of threads to decode with openMP on CPU
};

#endif
