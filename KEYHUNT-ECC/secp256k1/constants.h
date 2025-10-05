#pragma once
#include <stdint.h>

namespace kh_ecc {
namespace secp256k1 {

// secp256k1: y^2 = x^3 + 7 over Fp, a = 0, b = 7
// Prime field modulus p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// Group order n  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Base point G   = (Gx, Gy)

struct Params {
  // Little-endian 32-bit limbs (least-significant limb first)
  static constexpr uint32_t P[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
  };

  static constexpr uint32_t N[8] = {
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
  };

  // Curve coefficient a = 0 (Montgomery form uses a in Fp)
  static constexpr uint32_t A[8] = {
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
  };

  // Curve coefficient b = 7
  static constexpr uint32_t B[8] = {
    7u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
  };

  // Base point G (affine)
  static constexpr uint32_t Gx[8] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
  };
  static constexpr uint32_t Gy[8] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
  };

  // Montgomery parameters (for Base = uint32_t, 8 limbs)
  // pinv = -p^{-1} mod 2^32
  // p 的最低 limb 是 0xFFFFFC2F，其逆元的负数（mod 2^32）是 0xD2253531
  static constexpr uint32_t PINV = 0xD2253531u;
  // R = 2^256 mod p = 0x1000003d1 (little-endian)
  static constexpr uint32_t R[8]  = {0x000003D1u, 0x00000001u, 0u, 0u, 0u, 0u, 0u, 0u};
  // R2 = R^2 mod p = 0x1000007a2000e90a1 (little-endian)
  static constexpr uint32_t R2[8] = {0x000E90A1u, 0x000007A2u, 0x00000001u, 0u, 0u, 0u, 0u, 0u};
  static constexpr uint32_t ONE[8] = {1u,0u,0u,0u,0u,0u,0u,0u};
};

} // namespace secp256k1
} // namespace kh_ecc
