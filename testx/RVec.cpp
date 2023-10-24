#include "ROOT/RVec.hxx"
#include "Math/PtEtaPhiE4D.h"
#include "Math/Vector4D.h"
#include <assert.h>
#include <chrono>
#include <vector>

using arithmetic_type = double;
using vec4d =
    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<arithmetic_type>>;

template <class T> using Vector = ROOT::RVec<T>;

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

  Vector<arithmetic_type> eta1(N, 1.), eta2(N, 1.), phi1(N, 1.), phi2(N, 1.),
      pt1(N, 1.), pt2(N, 1.), mass1(N, 1.), mass2(N, 1.);

  Vector<arithmetic_type> masses =
      InvariantMasses(u_vectors, v_vectors, N, local_size);

  Vector<arithmetic_type> masses2 = ROOT::VecOps::InvariantMasses(pt1, eta1, phi1, mass1, pt2,
                                               eta2, phi2, mass2);

  assert((std::abs(masses[0] - (-2.3504)) <= 1e-5));
  std::cout << masses2[0] << std::endl;
  return 0;
}