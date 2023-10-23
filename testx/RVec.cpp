#include "Math/PtEtaPhiE4D.h"
#include "Math/Vector4D.h"
#include <chrono>
#include <vector>

using arithmetic_type = double;
using vec4d =
    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<arithmetic_type>>;
template <class T> using Vector = std::vector<T>;

Vector<arithmetic_type> InvariantMasses(const Vector<vec4d> v1,
                                        const Vector<vec4d> v2, const size_t N,
                                        const size_t local_size) {

  Vector<arithmetic_type> invMasses(N);

  auto start = std::chrono::system_clock::now();

  for (size_t i = 0; i < N; i++) {
    auto w = v1[i] + v2[i];
    invMasses[i] = w.mass();
  }

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "sycl time " << duration << " (s)" << std::endl;

  return invMasses;
}

Vector<vec4d> GenVectors(int n) {
  Vector<vec4d> vectors(n);

  // generate n -4 momentum quantities
  for (int i = 0; i < n; ++i) {
    // fill vectors
    vectors[i] = {1., 1., 1., 1.};
  }

  return vectors;
}

int main(int argc, char **argv) {

  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  size_t local_size = 128;

  auto u_vectors = GenVectors(N);
  auto v_vectors = GenVectors(N);

  Vector<arithmetic_type> masses =
      InvariantMasses(u_vectors, v_vectors, N, local_size);

  // std::cout << masses[0]<< std::endl;
  return 0;
}