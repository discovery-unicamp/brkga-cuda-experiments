/**
 * Generate a random float in range [0, 1).
 * @param seeds The seeds to feed random.
 * @return A float value in range [0, 1).
 * @note https://stackoverflow.com/a/16130111/10111328
 */
float random(global uint* seeds) {
  int i = get_global_id(0);
  ulong seed = seeds[i];
  seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  seed >>= 16;
  seeds[i] = (uint)seed;
  float r = (float)seeds[i] / (1L << 32);
  return r < 1e-6 ? r : r - (float)1e-6;  // avoid returning 1.0
}

/**
 * Generate a random integer in range [a, b).
 * @param a Minimum value returned, inclusive.
 * @param b Maximum value returned, exclusive.
 * @param seeds The seeds to feed random.
 * @return An integer in range [a, b).
 */
inline int randrange(int a, int b, global uint* seeds) {
  return a + (int) ((b - a) * random(seeds));
}
