#include "SYCLMath/Vector4D.h"
#include "SYCLMath/VecOps.h"
#include <chrono>
#include <vector>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


using arithmetic_type = double;
using vec4d = ROOT::Experimental::LorentzVector<
    ROOT::Experimental::PtEtaPhiM4D<arithmetic_type>>;
template <class T>
using Vector = std::vector<T>;



vec4d* GenVectors(int n)
{

  vec4d* vectors = new vec4d[n];

  // generate n -4 momentum quantities
  for (int i = 0; i < n; ++i)
  {
    // fill vectors
    vectors[i] = {1., 1., 1., 1.};
  }

  return vectors;
}

bool print_if_false(const bool assertion, size_t i) {
  if (!assertion) {
    std::cout << "Assertion failed at index "<< i << std::endl;
  }
  return assertion;
}

int main(int argc, char **argv)
{

  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  size_t local_size = 128;

  vec4d* u_vectors = GenVectors(N);
  vec4d* v_vectors = GenVectors(N);


  arithmetic_type* masses = ROOT::Experimental::InvariantMasses<arithmetic_type, vec4d>(u_vectors, v_vectors, N, local_size);

  for (size_t i=0; i<N; i++)
    assert(print_if_false((std::abs(masses[i] - 2.) <= 1e-5), i) );

  delete[] masses;
  return 0;
}
