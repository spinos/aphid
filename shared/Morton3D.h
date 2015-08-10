#ifndef MORTON3D_H
#define MORTON3D_H

#define min(a, b) (a < b?a: b)
#define max(a, b) (a > b?a: b)

// http://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit
inline unsigned expandBits(unsigned v) 
{ 
    // v = (v * 0x00010001u) & 0xFF0000FFu; 
    // v = (v * 0x00000101u) & 0x0F00F00Fu; 
    // v = (v * 0x00000011u) & 0xC30C30C3u; 
    // v = (v * 0x00000005u) & 0x49249249u; 
    
    v = (v | v << 16) & 0xFF0000FFu; 
    v = (v | v << 8) & 0x0F00F00Fu; 
    v = (v | v << 4) & 0xC30C30C3u; 
    v = (v | v << 2) & 0x49249249u; 
    return v; 
}

// Calculates a 30-bit Morton code for the 
// given 3D point located within the unit cube [0,1].
inline unsigned encodeMorton3D(unsigned x, unsigned y, unsigned z) 
{ 
    x = min(max(x, 0), 1023); 
    y = min(max(y, 0), 1023); 
    z = min(max(z, 0), 1023); 
    unsigned xx = expandBits((unsigned)x); 
    unsigned yy = expandBits((unsigned)y); 
    unsigned zz = expandBits((unsigned)z); 
    return xx << 2 | yy << 1 | zz; 
}

// decoding morton code to cartesian coordinate
// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/

inline unsigned Compact1By2(unsigned x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
  return x;
}

inline void decodeMorton3D(unsigned code, unsigned & x, unsigned & y, unsigned & z)
{
    x = Compact1By2(code >> 2);
    y = Compact1By2(code >> 1);
    z = Compact1By2(code >> 0);
}

// // http://stackoverflow.com/questions/4909263/how-to-efficiently-de-interleave-bits-inverse-morton
// uint64_t morton3(uint64_t x) {
    // x = x & 0x9249249249249249;
    // x = (x | (x >> 2))  & 0x30c30c30c30c30c3;
    // x = (x | (x >> 4))  & 0xf00f00f00f00f00f;
    // x = (x | (x >> 8))  & 0x00ff0000ff0000ff;
    // x = (x | (x >> 16)) & 0xffff00000000ffff;
    // x = (x | (x >> 32)) & 0x00000000ffffffff;
    // return x;
// }
// uint64_t bits; 
// uint64_t x = morton3(bits)
// uint64_t y = morton3(bits>>1)
// uint64_t z = morton3(bits>>2)

#endif        //  #ifndef MORTON3D_H

