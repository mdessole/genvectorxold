#include <iostream>

#include "SYCLMath/Cartesian2D.h"
#include "SYCLMath/Polar2D.h"
#include "SYCLMath/Vector2D.h"

#include "Math/Cartesian2D.h"
#include "Math/Vector2D.h"

// using namespace ROOT::Math;
using vec2d = ROOT::Experimental::DisplacementVector2D<
    ROOT::Experimental::Cartesian2D<double>>;
using vec2dR =
    ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double>>;

template <class V> void print_vec2d(V v, const char *name) {
  std::cout << name << " : " << v.x() << ", " << v.y() << std::endl;
}

int main() {
  // Cartesian2D point2dA{1.0,2.0}, point2dB{-1.0,-2.0}, point2dC;
  vec2d u{1.0, 2.0}, v{-2.0, 2.0}, w{};
  vec2dR Ru{1.0, 2.0}, Rv{-2.0, 2.0}, Rw{};

  print_vec2d(u, "u");
  print_vec2d(v, "v");
  print_vec2d(w, "w");
  print_vec2d(Ru, "Ru");
  print_vec2d(Rv, "Rv");
  print_vec2d(Rw, "Rw");
}