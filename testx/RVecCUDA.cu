#include "SYCLMath/Vector4D.h"
#include <chrono>
#include <vector>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


using arithmetic_type = double;
using vec4d = ROOT::Experimental::LorentzVector<
    ROOT::Experimental::PtEtaPhiM4D<arithmetic_type>>;
template <class T>
using Vector = std::vector<T>;

#ifndef RVecCUDA_H
#define RVecCUDA_H

// namespace ROOT {
// namespace Experimental {

template <class Vec, class Mass>
__global__ void InvariantMassKernel(Vec vec, Mass m, size_t N)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N)
  {
    vec4d w = vec[id];
    m[id] = w.mass();
  }
}

template <class Vec, class Mass>
__global__ void InvariantMassesKernel(Vec v1, Vec v2, Mass m, size_t N)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N)
  {
    vec4d w = v1[id];
    w+=v2[id];
    m[id] = w.mass();
  }
}

//} // namespace Experimental
//} // namespace ROOT
#endif

arithmetic_type* InvariantMasses(vec4d* v1, vec4d* v2,  const size_t N,
                                      const size_t local_size)
{

  arithmetic_type* invMasses = new arithmetic_type[N];

  size_t sizeVec = N*sizeof(vec4d);
  auto start = std::chrono::system_clock::now();
  cudaError_t err;
{
// Allocate the device input vector A
  vec4d* d_v1 = NULL;
  err = cudaMalloc((void **)&d_v1, sizeVec);


  // Allocate the device input vector B
  vec4d* d_v2 = NULL;
  err = cudaMalloc((void **)&d_v2, sizeVec);

  // Allocate the device output vector C
  arithmetic_type* d_invMasses = NULL;
  err = cudaMalloc((void **)&d_invMasses, sizeVec);

  cudaMemcpy ( d_v1, v1, sizeVec, cudaMemcpyHostToDevice );
  cudaMemcpy ( d_v2, v2, sizeVec, cudaMemcpyHostToDevice );

  InvariantMassesKernel<<<fmax(1, N / local_size), local_size>>>(d_v1, d_v2, d_invMasses, N);

  cudaMemcpy ( invMasses, d_invMasses, sizeVec, cudaMemcpyDeviceToHost );
 

  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_invMasses);

}
 
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "cuda time " << duration << " (s)" << std::endl;


  return invMasses;
}

arithmetic_type* InvariantMass(vec4d* v1, const size_t N, const size_t local_size)
{

  arithmetic_type* invMasses = new arithmetic_type[N];

  size_t sizeVec = N*sizeof(vec4d);
  auto start = std::chrono::system_clock::now();
  cudaError_t err;
{
// Allocate the device input vector A
  vec4d* d_v1 = NULL;
  err = cudaMalloc((void **)&d_v1, sizeVec);



  // Allocate the device output vector C
  arithmetic_type* d_invMasses = NULL;
  err = cudaMalloc((void **)&d_invMasses, sizeVec);

  cudaMemcpy ( d_v1, v1, sizeVec, cudaMemcpyHostToDevice );


  InvariantMassKernel<<<fmax(1, N / local_size), local_size>>>(d_v1, d_invMasses, N);

  cudaMemcpy ( invMasses, d_invMasses, sizeVec, cudaMemcpyDeviceToHost );
 
  cudaFree(d_v1);
  cudaFree(d_invMasses);

}
 
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "cuda time " << duration << " (s)" << std::endl;


  return invMasses;
}

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

int main(int argc, char **argv)
{

  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  size_t local_size = 128;

  vec4d* u_vectors = GenVectors(N);
  vec4d* v_vectors = GenVectors(N);


  arithmetic_type* masses = InvariantMasses(u_vectors, v_vectors, N, local_size);

//InvariantMass(u_vectors, N, local_size);
  //    InvariantMasses(u_vectors, v_vectors, N, local_size);

  //std::cout << "masses[0] " << masses[5] << std::endl;
  assert((std::abs(masses[0] - 2.) <= 1e-5));
  return 0;
}
