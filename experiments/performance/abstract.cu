#include "aaa.cuh"

struct Chromosome {
  __host__ __device__ Chromosome(float* _population, uint _length, uint _k)
      : population(_population), length(_length), k(_k) {}

  __host__ __device__ virtual float operator[](uint index) = 0;

  float* population;
  uint length;
  uint k;
};

// The chromosome is on a row of the matrix
struct ChromosomeRowFormat : public Chromosome {
  __host__ __device__ ChromosomeRowFormat(float* _population,
                                          uint _length,
                                          uint _k)
      : Chromosome(_population, _length, _k) {}

  __host__ __device__ virtual float operator[](uint index) override {
    return this->population[this->k * this->length + index];
  }
};

__global__ void first(ChromosomeRowFormat cr, float* value) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) *value = cr[0];
}

int main() {
  // random values
  const uint n = 107;
  const uint len = 35;
  const uint row = 3;
  const float expected = 1298.34765;

  vector<float> chromosomes(n * len);
  chromosomes[row * len] = expected;

  float* dChromosomes = nullptr;
  check(cudaMalloc(&dChromosomes, n * len * sizeof(float)));
  check(cudaMemcpy(dChromosomes, chromosomes.data(), n * len * sizeof(float),
                   cudaMemcpyHostToDevice));

  float* dValue = nullptr;
  check(cudaMalloc(&dValue, sizeof(float)));

  first<<<10, 10>>>(ChromosomeRowFormat(dChromosomes, len, row), dValue);
  check(cudaDeviceSynchronize());

  float value = -1;
  check(cudaMemcpy(&value, dValue, sizeof(float), cudaMemcpyDeviceToHost));

  cout << fixed << setprecision(9) << expected << ' ' << value << endl;

  check(cudaFree(dChromosomes));
  check(cudaFree(dValue));

  return 0;
}
