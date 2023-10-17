#include <iostream>

#include "Math/Cartesian2D.h"
#include "Math/Polar2D.h"
#include "Math/Vector2D.h"

// using namespace ROOT::Math;
using vec2d = ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double>>;

void print_vec2d(vec2d v, const char *name) {
  std::cout << name << " : " << v.x() << ", " << v.y() << std::endl;
}

int main() {
  // Cartesian2D point2dA{1.0,2.0}, point2dB{-1.0,-2.0}, point2dC;
  vec2d u{1.0, 2.0}, v{-2.0, 2.0}, w{};

  print_vec2d(u, "u");
  print_vec2d(v, "v");
  print_vec2d(w, "w");
}