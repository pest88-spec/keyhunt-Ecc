#pragma once
#include <stdint.h>
#include "fp_montgomery.h"

namespace kh_ecc {
namespace secp256k1 {

struct Affine {
  Fp x;
  Fp y;

  __device__ __forceinline__ static Affine zero() {
    Affine r; r.x = Fp::mont_one(); r.y = Fp::zero(); return r;
  }
  __device__ __forceinline__ bool is_zero() const { return y.is_zero(); }
};

struct ECPointJacobian {
  using BaseField = Fp;

  Fp x;
  Fp y;
  Fp z; // z == 0 => point at infinity

  __device__ __forceinline__ static ECPointJacobian zero() {
    ECPointJacobian r; r.x = Fp::mont_one(); r.y = Fp::mont_one(); r.z = Fp::zero(); return r;
  }
  __device__ __forceinline__ bool is_zero() const { return z.is_zero(); }

  __device__ __forceinline__ static void set_zero(ECPointJacobian &o) { o = zero(); }

  __device__ __forceinline__ Affine to_affine() const {
    if (is_zero()) return Affine::zero();
    // inv_z, inv_z2, inv_z3
    Fp inv_z  = z.inverse();
    Fp inv_z2 = inv_z.square();
    Fp inv_z3 = inv_z2 * inv_z;
    Affine r;
    r.x = x * inv_z2;
    r.y = y * inv_z3;
    return r;
  }

  // a = 0 doubling (dbl-2007-bl)
  __device__ __forceinline__ ECPointJacobian dbl() const {
    if (is_zero()) return *this;
    ECPointJacobian R;
    Fp XX = x.square();        // XX = X1^2
    Fp YY = y.square();        // YY = Y1^2
    Fp YYYY = YY.square();     // YYYY = YY^2
    Fp S = x * YY;             // S = X1*YY
    S = S + S;                 // S = 2*X1*YY
    S = S + S;                 // S = 4*X1*YY (修正!)
    Fp M = XX + XX + XX;       // M = 3*XX
    Fp T = M.square();         // T = M^2
    Fp twoS = S + S;           // 2*S
    R.x = T - twoS;            // X3 = T - 2*S
    Fp S_minus_X3 = S - R.x;   // S - X3
    Fp eightYYYY = YYYY + YYYY;  // 2*YYYY
    eightYYYY = eightYYYY + eightYYYY; // 4*YYYY
    eightYYYY = eightYYYY + eightYYYY; // 8*YYYY
    R.y = M * S_minus_X3 - eightYYYY;  // Y3 = M*(S - X3) - 8*YYYY
    Fp twoY = y + y;           // 2*Y1
    R.z = twoY * z;            // Z3 = 2*Y1*Z1
    return R;
  }

  // general add (add-2007-l variant), a = 0
  __device__ __forceinline__ ECPointJacobian add(const ECPointJacobian &o) const {
    if (o.is_zero()) return *this;
    if (is_zero()) return o;

    Fp Z1Z1 = z.square();
    Fp Z2Z2 = o.z.square();
    Fp U1 = x * Z2Z2;
    Fp U2 = o.x * Z1Z1;
    Fp S1 = y * o.z * Z2Z2;   // y1 * z2 * z2^2 = y1 * z2^3
    Fp S2 = o.y * z * Z1Z1;

    if (U1 == U2) {
      if (!(S1 == S2)) {
        return zero();
      }
      return dbl();
    }

    Fp H = U2 - U1;
    Fp Rr = S2 - S1;
    Fp HH = H.square();
    Fp HHH = H * HH;
    Fp V = U1 * HH;

    ECPointJacobian R;
    R.x = Rr.square() - HHH - (V + V);
    R.y = Rr * (V - R.x) - S1 * HHH;
    R.z = z * o.z * H;  // 修正: Z3 = Z1 * Z2 * H (不是 2*Z1*Z2*H)
    return R;
  }

  // mixed add (Jacobian + Affine), a = 0
  __device__ __forceinline__ ECPointJacobian mixed_add(const Affine &o) const {
    if (o.is_zero()) return *this;
    if (is_zero()) { ECPointJacobian r; r.x = o.x; r.y = o.y; r.z = Fp::mont_one(); return r; }

    Fp Z1Z1 = z.square();
    Fp U2 = o.x * Z1Z1;
    Fp S2 = o.y * z * Z1Z1; // y2 * z1 * z1^2 = y2 * z1^3

    if (x == U2) {
      if (!(y == S2)) {
        return zero();
      }
      return dbl();
    }

    Fp H = U2 - x;
    Fp Rr = S2 - y;
    Fp HH = H.square();
    Fp HHH = H * HH;
    Fp V = x * HH;

    ECPointJacobian R;
    R.x = Rr.square() - HHH - (V + V);
    R.y = Rr * (V - R.x) - y * HHH;
    R.z = z * H; // mixed: Z3 = Z1 * H
    return R;
  }

  __device__ __forceinline__ ECPointJacobian operator+(const ECPointJacobian &o) const {
    if (o.is_zero()) return *this;
    if (is_zero()) return o;
    return add(o);
  }

  __device__ __forceinline__ ECPointJacobian operator+(const Affine &o) const {
    if (o.is_zero()) return *this;
    if (is_zero()) { ECPointJacobian r; r.x = o.x; r.y = o.y; r.z = Fp::mont_one(); return r; }
    return mixed_add(o);
  }
};

} // namespace secp256k1
} // namespace kh_ecc
