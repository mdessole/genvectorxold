#include "SYCLMath/Vector4D.h"
#include "SYCLMath/VecOps.h"
#include <chrono>
#include <vector>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#ifdef SINGLE_PRECISION
using arithmetic_type = float;
#else
using arithmetic_type = double;
#endif

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

#ifdef ROOT_MEAS_TIMING
      std::cout<< "ROOT_MEAS_TIMING defined \n"; 
#endif
#ifdef ROOT_MATH_SYCL
      std::cout<< "ROOT_MATH_SYCL defined \n"; 
#endif
#ifdef ROOT_MATH_CUDA
      std::cout<< "ROOT_MATH_CUDA defined \n"; 
#endif
#ifdef SINGLE_PRECISION
      std::cout<< "SINGLE_PRECISION defined \n"; 
#endif
  int count;
  cudaGetDeviceCount(&count); 
  std::cout << "CUDA available devices: " << count << std::endl;
  cudaSetDevice(0);
  //cudaInitDevice(0,);

  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  std::string arg2 = argv[2];
  std::size_t nruns = std::stoi(arg2, &pos);
  size_t local_size = 128;

  vec4d* u_vectors = GenVectors(N);
  vec4d* v_vectors = GenVectors(N);

  arithmetic_type* masses =  new arithmetic_type[N];

  for (size_t i=0; i<nruns; i++)
    masses = ROOT::Experimental::InvariantMasses<arithmetic_type, vec4d>(u_vectors, v_vectors, N, local_size);

  for (size_t i=0; i<N; i++)
    assert(print_if_false((std::abs(masses[i] - 2.) <= 1e-5), i) );

  delete[] u_vectors;
  delete[] v_vectors;
  delete[] masses;
  return 0;
}
