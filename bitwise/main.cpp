/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 1/11/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>
#include <boost/format.hpp>
#ifdef __APPLE__
typedef unsigned long long uint64;
#else
typedef unsigned long uint64;
#endif

typedef unsigned int uint;

#define min(a, b) (a < b?a: b)
#define max(a, b) (a > b?a: b)

uint64 upsample(uint a, uint b) 
{ return ((uint64)a << 32) | (uint64)b; }

uint expandBits(uint v) 
{ 
    v = (v * 0x00010001u) & 0xFF0000FFu; 
    v = (v * 0x00000101u) & 0x0F00F00Fu; 
    v = (v * 0x00000011u) & 0xC30C30C3u; 
    v = (v * 0x00000005u) & 0x49249249u; 
    return v; 
}

// Calculates a 30-bit Morton code for the 
// given 3D point located within the unit cube [0,1].
uint morton3D(uint x, uint y, uint z) 
{ 
    x = min(max(x, 0.0f), 1023); 
    y = min(max(y, 0.0f), 1023); 
    z = min(max(z, 0.0f), 1023); 
    uint xx = expandBits((uint)x); 
    uint yy = expandBits((uint)y); 
    uint zz = expandBits((uint)z); 
    return xx * 4 + yy * 2 + zz; 
}

const char *byte_to_binary(uint x)
{
    static char b[33];
    b[32] = '\0';

    for (int z = 0; z < 32; z++) {
        b[31-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

    return b;
}

const char *byte_to_binary64(uint64 x)
{
    static char b[65];
    b[64] = '\0';

    for (int z = 0; z < 64; z++) {
        b[63-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

    return b;
}

// the number, between 0 and 64 inclusive, of consecutive zero bits 
// starting at the most significant bit (i.e. bit 63) of 64-bit integer parameter x. 
int clz(uint64 x)
{
	static char b[64];
	int z;
    for (z = 0; z < 64; z++) {
        b[63-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

	int nzero = 0;
	for (z = 0; z < 64; z++) {
		if(b[z] == '0') nzero++;
		else break;
    }
	return nzero;
}


int main(int argc, char * const argv[])
{
	std::cout<<"bitwise test\n";
	std::cout<<"bit size of uint:   "<<sizeof(uint) * 8<<"\n";
	std::cout<<"bit size of uint64: "<<sizeof(uint64) * 8<<"\n";
	
	int x0 = 1023, y0 = 0, z0 = 8;
	int x1 = 900, y1 = 116, z1 = 7;
	uint m0 = morton3D(x0, y0, z0);
	uint m1 = morton3D(x1, y1, z1);
	uint64 u0 = upsample(m0, 8);
	uint64 u1 = upsample(m1, 9);
	std::cout<<boost::format("morton code of (%1%, %2%, %3%): %4%\n") % x0 % y0 % z0 % byte_to_binary(m0);
	std::cout<<boost::format("morton code of (%1%, %2%, %3%): %4%\n") % x1 % y1 % z1 % byte_to_binary(m1);
	std::cout<<boost::format("upsample m0: %1%\n") % byte_to_binary64(u0);
	std::cout<<boost::format("upsample m1: %1%\n") % byte_to_binary64(u1);
	std::cout<<boost::format("upsample m0 ^ upsample m1: %1%\n") % byte_to_binary64(u0 ^ u1);
	std::cout<<boost::format("common prefix length: %1%\n") % clz(u0 ^ u1);
	uint64 bitMask = ((uint64)(~0)) << (64 - clz(u0 ^ u1));
	std::cout<<boost::format("bit mask: %1%\n") % byte_to_binary64(bitMask);
	uint64 sharedBits = u0 & u1;
	std::cout<<boost::format("shared bits: %1%\n") % byte_to_binary64(sharedBits);
	std::cout<<boost::format("common prefix: %1%\n") % byte_to_binary64(sharedBits & bitMask);
	
	std::cout<<"end of test\n";
	return 0;
}