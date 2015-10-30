/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 1/11/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>
#include <map>
//#include <boost/timer/timer.hpp>
#include <boost/format.hpp>
#include <boost/timer.hpp>

#include <tetrahedron_math.h>
#include <MersenneTwister.h>
#include <AOrientedBox.h>
#include <IndexArray.h>
#include <SHelper.h>
#include <KdNTree.h>
#include <KdBuilder.h>

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

void downsample(uint64 combined, uint & a, uint & b)
{
    a = combined >> 32;
    b = combined & ~0x80000000;
}

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

struct Stripe {
    char rgba[2048]; // 256 * 4 * 2
    float z[256];
};

// decoding morton code to cartesian coordinate
// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/

uint Compact1By2(uint x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
  return x;
}

uint DecodeMorton3Z(uint code)
{
  return Compact1By2(code >> 0);
}

uint DecodeMorton3Y(uint code)
{
  return Compact1By2(code >> 1);
}

uint DecodeMorton3X(uint code)
{
  return Compact1By2(code >> 2);
}

void testArray()
{
    const int a[3][3][3] = {{{0,1,2},{3,4,5},{6,7,8}},
        {{9,10,11},{12,13,14},{15,16,17}},
        {{18,19,20},{21,22,23},{24,25,26}}};
    
    int i, j, k;
    for(k=0;k<3;k++) {
        for(j=0;j<3;j++) {
            for(i=0;i<3;i++) {
                std::cout<<"a["<<k<<"]["<<j<<"]["<<i<<"] "<<a[k][j][i]<<"\n";
            }
        }
    }
}

void testNan()
{
#ifdef TEST_NAN
    float a = 0.f/0.f;
    std::cout<<boost::format("zero divided by zero: %1%\n") % a;
	std::cout<<boost::format("is nan: %1%\n") % (a == a);
#endif
}

void testInf()
{
#ifdef TEST_NAN
    float a = 1.f/0.f;
    std::cout<<boost::format("one divided by zero: %1%\n") % a;
	std::cout<<boost::format("is nan: %1%\n") % (a > 1e38);
	a = -1.f/0.f;
    std::cout<<boost::format("negative one divided by zero: %1%\n") % a;
	std::cout<<boost::format("is nan: %1%\n") % (a < -1e38);
#endif
}

void testTetrahedronDegenerate()
{
    Vector3F p[4];
    p[0].set(0.f, 1.f, 0.f);
    p[1].set(.1f, 1.f, 0.f);
    p[2].set(.1f, 1.f, .1f);
    p[3].set(0.f, 1.1f, 0.f);
    
    Vector3F e1 = p[1]-p[0];
	Vector3F e2 = p[2]-p[0];
	Vector3F e3 = p[3]-p[0];
	const float sc = e1.length() * e2.length() * e3.length() / 6.f;
	std::cout<<"sc "<<sc<<"\n";
	
    Matrix44F mat;
    std::cout<<"det tet "<<determinantTetrahedron(mat, p[0], p[1], p[2], p[3])<<"\n";
    std::cout<<" volume "<<tetrahedronVolume(p)<<"\n";
}

void testKij()
{
    int k = 199;
    int i = 2;
    int j = 3;
    int c = (k<<5 | ( i<<3 | j));
    std::cout<<boost::format("kij    %1%\n") % byte_to_binary(c);	
    std::cout<<boost::format("k %1%\n") % (c>>5);
    std::cout<<boost::format("mask   %1%\n") % byte_to_binary(31);
    std::cout<<boost::format("masked %1%\n") % byte_to_binary(c&31);
    std::cout<<boost::format("masked>>3 %1%\n") % byte_to_binary((c&31)>>3);
    std::cout<<boost::format("i         %1%\n") % ((c&31)>>3);
    std::cout<<boost::format("mask   %1%\n") % byte_to_binary(3);
    std::cout<<boost::format("masked  %1%\n") % byte_to_binary(c&3);
    std::cout<<boost::format("j         %1%\n") % (c&3);	
}

void testKijt()
{
    int k = 11110;
    int i = 3;
    int j = 3;
    int t = 0;
    int c = ( k<<5 | ( i<<3 | (j<<1 | t) ) );
    std::cout<<boost::format("kijt    %1%\n") % byte_to_binary(c);	
    std::cout<<boost::format("k %1%\n") % (c>>5);
    std::cout<<boost::format("masked>>3 %1%\n") % byte_to_binary((c&31)>>3);
    std::cout<<boost::format("i         %1%\n") % ((c&31)>>3);
    std::cout<<boost::format("j         %1%\n") % ((c&7)>>1);
    std::cout<<boost::format("t         %1%\n") % (c&1);
    	
}

void testMap()
{
    std::cout<<" test str map\n";
    std::map<std::string, int > pool;
    pool["|group2|group1|pCube"] = 0;
    pool["|group2"] = 1;
    pool["|group2|pSphere"] = 2;
    pool["|helix"] = 3;
    pool["|group1|pCube"] = 4;
    
    std::map<std::string, int >::const_iterator it = pool.begin();
    for(;it!=pool.end();++it)
        std::cout<<"\n "<<it->second<<" "<<it->first;
    std::cout<<" \n";
}

void testCell()
{
    std::cout<<"\n test cell\n";
    Vector3F origin;
    std::cout<<"\n origin "<<origin;
    float span = 1023.1276f;
    float h = span / 1024.f;
    float ih = 1.f / h;
    std::cout<<"\n span "<<span;
    Vector3F p(32.5468f, 33.25f, 37.76f);
    std::cout<<"\n p "<<p;
    unsigned x = (p.x - origin.x) * ih;
    unsigned y = (p.y - origin.y) * ih;
    unsigned z = (p.z - origin.z) * ih;
    std::cout<<"\n level 10 xyz"<<x<<","<<y<<","<<z;
    
    int level = 7;
    int d = 10 - level;
    x = x>>d;
    y = y>>d;
    z = z>>d;
    x = x<<d;
    y = y<<d;
    z = z<<d;
    
    std::cout<<"\n level 7 origin xyz"<<x<<","<<y<<","<<z;
    
    int a = (1<<(d-1)) - 1;
    std::cout<<"\n level 7 half "<<a;
    
    x += a;
    y += a;
    z += a;
    std::cout<<"\n level 7 center xyz"<<x<<","<<y<<","<<z;
}

void testMersenne()
{
	std::cout<<"\n test Mersenne Twister pseudorandom number generator\n";
	
	MersenneTwister rng(1);
	int i = 0;
	for(;i<255;i++) {
		std::cout<<" "<<rng.iRandom(0, 1000);
	}
}

void testOBox()
{
    std::cout<<"\n test oriented box ";
    std::vector<AOrientedBox> vecbox;
    int n = 65536;
    int i, j;
    
    int * ind = new int[n];
    MersenneTwister rng(1);
    for(i=0;i<n;i++) ind[i] = rng.iRandom(0, 65535);
    
    boost::timer bTimer;
	bTimer.restart();
    for(i=0;i<n;i++) vecbox.push_back(AOrientedBox());
    std::cout << "\n create vec took " << bTimer.elapsed()<<" seconds";
	
    bTimer.restart();
    for(j=0;j<100;j++) {
        for(i=0;i<n;i++) {
            AOrientedBox * a = &vecbox[ind[i]];
            Vector3F d = a->extent();
        }
    }
    std::cout << "\n visit vec took " << bTimer.elapsed()<<" seconds";
}

void testTimedArray(unsigned n, unsigned nrep)
{
    unsigned i,j, b;
    boost::timer bTimer;
    
    IndexArray arrs;
	//arrs.expandBy(n);
	for(i=0;i<n;i++) {
		arrs.expandBy(1);
		//*arrs.asIndex() = i;
		arrs.setValue(i);
		arrs.next();
	}
    std::cout << "\n fill "<<n<< " array took " << bTimer.elapsed() <<" secs";

    bTimer.restart();
    
    for(j=0;j<nrep;j++) {
        b= 0;
        arrs.begin();
        for(i=0;i<n;i++) {
            //b += *arrs.asIndex();
            b += arrs.value();
            arrs.next();
        }
    }
    std::cout << "\n access "<<n<< " array took " << bTimer.elapsed() <<" secs";
	std::cout<<"\n sum "<<b;
    unsigned access = *arrs.asIndex(9235);
	std::cout<<" test access [9235] "<<(access);
}

void testTimedVec(unsigned n, unsigned nrep)
{
    boost::timer bTimer;
    bTimer.restart();
    unsigned i, j, b;
    std::vector<unsigned> vecs;
	//vecs.resize(n);
	for(i=0;i<n;i++) //vecs[i] = i;
        vecs.push_back(i);
    
    std::cout << "\n fill "<<n<< " vector took " << bTimer.elapsed() <<" secs";
	
    bTimer.restart();
    
	for(j=0;j<nrep;j++) {
        b= 0;
        for(i=0;i<n;i++) b += vecs[i];
    }
	std::cout << "\n access "<<n<< " vector took " << bTimer.elapsed() <<" secs";
	std::cout<<"\n sum "<<b;
}

void testTimedMem(unsigned n, unsigned nrep)
{
    boost::timer bTimer;
    bTimer.restart();
    unsigned i, j, b;
    unsigned * mems = new unsigned[n];
	for(i=0;i<n;i++) mems[i] = i;
    
	std::cout << "\n fill "<<n<< " mem took " << bTimer.elapsed() <<" secs";
	
    bTimer.restart();
	for(j=0;j<nrep;j++) {
        b= 0;
        for(i=0;i<n;i++) b += mems[i];
    }
	std::cout << "\n access "<<n<< " mem took " << bTimer.elapsed() <<" secs";
    std::cout<<"\n sum "<<b;
}

void testVecArray()
{
	std::cout<<"\n test array";
	const unsigned n = 9999999;
    
    testTimedMem(n, 100);
    testTimedVec(n, 100);
    testTimedArray(n, 100);
}

void testFind()
{
	std::cout<<" find translateX in /a/b/b/abc_translateX "<<SHelper::Find("/a/b/b/abc_Lcl Translate/X", "lcl translate/X", true);
}

class TestBox : public BoundingBox
{
public:
    TestBox() {}
    virtual ~TestBox() {}
    BoundingBox calculateBBox() const
    { return * this; }
    BoundingBox bbox() const
    { return * this; }
};

void testTree()
{
    std::cout<<" test kdtree\n";
	const int n = 1100;
    SahSplit<TestBox> su(n);
	BoundingBox rootBox;
    int i;
    for(i=0; i<n; i++) {
        TestBox *a = new TestBox;
        float r = float( rand() % 999 ) / 999.f;
        float th = float( rand() % 999 ) / 999.f * 1.5f;
        float x = 20.f + 40.f * r * cos(th);
        float y = 20.f + 32.f * r * sin(th);
        float z = -40.f + 4.f * float( rand() % 999 ) / 999.f;
        a->setMin(-1 + x, -1 + y, -1 + z);
        a->setMax( 1 + x,  1 + y,  1 + z);
        su.set(i, a);
		rootBox.expandBy(a->calculateBBox());
    }
	std::cout<<"\n root box "<<rootBox;
	su.setBBox(rootBox);
	
	int maxLevel = 1;
	int nn = 1<<KdNode4::BranchingFactor();
	while(nn < n>>KdNode4::BranchingFactor()-1) {
		std::cout<<" nn "<<nn;
		nn = nn<<KdNode4::BranchingFactor();
		maxLevel++;
	}
	std::cout<<"\n nn "<<nn<<" calc tr max level "<<maxLevel;
	
    KdNTree<TestBox, KdNode4 > tr(maxLevel, n);
	
	std::cout<<" max n nodes "<<tr.maxNumNodes();
	
	KdNBuilder<4, 4, TestBox, KdNode4 > bud;
	bud.build(&su, tr.nodes(), tr.root());
}

void testRgba()
{
    unsigned mred = 0xff000000;
    std::cout<<boost::format("mask 0xff000000: %1%\n") % byte_to_binary(mred);
    unsigned mgreen = 0x00ff0000;
    std::cout<<boost::format("mask 0x00ff0000: %1%\n") % byte_to_binary(mgreen);
    unsigned mblue = 0x0000ff00;
    std::cout<<boost::format("mask 0x0000ff00: %1%\n") % byte_to_binary(mblue);
    unsigned ma = 0x000000ff;
    std::cout<<boost::format("mask 0x000000ff: %1%\n") % byte_to_binary(ma);
    unsigned a = (255<<24) | (0<<16) | (99<<8) | 1;
    std::cout<<boost::format("       rgba: %1%\n") % byte_to_binary(a);
    std::cout<<boost::format("rgba masked: %1%\n") % byte_to_binary(a & mred);
    unsigned rr = (a & mred)>>24;
    unsigned rg = (a & mgreen)>>16;
    unsigned rb = (a & mblue)>>8;
    unsigned ra = (a & ma);
    std::cout<<boost::format("recovered rgba: %1% %2% %3% %4%\n") % rr % rg % rb % ra;
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
	std::cout<<boost::format("mask 0x80000000: %1%\n") % byte_to_binary(0x80000000);
	std::cout<<boost::format("~mask ~0x80000000: %1%\n") % byte_to_binary(~0x80000000);
	std::cout<<boost::format("8 masked: %1%\n") % byte_to_binary(8 | 0x80000000);
	std::cout<<boost::format("8 unmasked: %1%\n") % byte_to_binary(8 & (~0x80000000));
	std::cout<<boost::format("12: %1%\n") % byte_to_binary(12);
	std::cout<<boost::format("12 left 24 masked: %1%\n") % byte_to_binary(12<<24 | 0x80000000);
	std::cout<<boost::format("12 right 24 unmasked: %1%\n") % byte_to_binary(((12<<24 | 0x80000000)&(~0x80000000))>>24);
	
	std::cout<<boost::format("12 left 24 with 1923 masked: %1%\n") % byte_to_binary((12<<24 | 1923) | 0x80000000);
	std::cout<<boost::format("12 right 24 with 1923 unmasked: %1%\n") % byte_to_binary((((12<<24 | 1923) | 0x80000000)&(~0x80000000))>>24);
	std::cout<<boost::format("id masked: %1%\n") % byte_to_binary(~0x800000);
	std::cout<<boost::format("1923: %1%\n") % byte_to_binary(1923);
	std::cout<<boost::format("<<7 >>7: %1%\n") % byte_to_binary((((12<<24 | 1923) | 0x80000000)<<7)>>7);
	
	std::cout<<boost::format("sizeof Stripe: %1%\n") % sizeof(std::map<int, Stripe *>);
	
	int a[8] = {99, 83, -73, -32, 48, 1, 53, -192};
	int b[4];
	memcpy(b, &a[4], 4 * 4);

	int i;
	for(i=0; i<4; i++) {
	    std::cout<<boost::format("b[%1%] %2%\n") % i % b[i]; 
	}
	
	float c[4];
	for(i=0; i<4; i++) {
	    std::cout<<boost::format("c[%1%] %2%\n") % i % c[i]; 
	}
// If n is a power of 2, 
// (i/n) is equivalent to (i>>log2(n)) 
// and (i%n) is equivalent to (i&(n-1)); 	
	std::cout<<boost::format("1023 & 255 %1%\n") % (1023 & 255);
	std::cout<<boost::format("1024 & 255 %1%\n") % (1024 & 255);
	std::cout<<boost::format("1023>>10 %1%\n") % (1023>>10);
	std::cout<<boost::format("3<<2 %1%\n") % (3<<2);
	std::cout<<boost::format("5>>2 %1%\n") % (5>>2);
	std::cout<<boost::format("513 / 256 %1%\n") % ((513>>8) + ((513 & 255) != 0));
	
	x1 = 99, y1 = 736, z1 = 121;
	m1 = morton3D(x1, y1, z1);
	
	std::cout<<boost::format("morton code of (%1%, %2%, %3%): %4%\n") % x1 % y1 % z1 % byte_to_binary(m1);	
	std::cout<<boost::format("decode morton code to xyz: (%1%, %2%, %3%)\n") % DecodeMorton3X(m1) % DecodeMorton3Y(m1) % DecodeMorton3Z(m1);
		
	std::cout<<boost::format("mask of last32 bit: %1%\n") % byte_to_binary64(~0x80000000);
	
	uint upa = 892500748;
	uint upb = 1219402334;
	uint64 upc = upsample(upa, upb);
	std::cout<<boost::format("upsample( %1%, %2% ): %3%\n") % upa % upb % upc;
	downsample(upc, upa, upb);
	std::cout<<boost::format("downsample( %1% ): %2% %3%\n") % upc % upa % upb;
	
	testArray();
	testNan();
	testInf();
	testTetrahedronDegenerate();
	testKijt();
	
	m1 = 22860495;
	std::cout<<boost::format("22860495: %1%\n") % byte_to_binary(m1);
	std::cout<<boost::format("decode morton code to xyz: (%1%, %2%, %3%)\n") % DecodeMorton3X(m1) % DecodeMorton3Y(m1) % DecodeMorton3Z(m1);
	
	m1 = 802499201;
	std::cout<<boost::format("802499201: %1%\n") % byte_to_binary(m1);
	std::cout<<boost::format("decode morton code to xyz: (%1%, %2%, %3%)\n") % DecodeMorton3X(m1) % DecodeMorton3Y(m1) % DecodeMorton3Z(m1);
	
	std::cout<<boost::format("39 mod 16: %1%\n") % (39 & 15);
	
	// testVecArray();
	
	// testMersenne();
    
    // testOBox();
	// testRgba();
    testTree();
    
	std::cout<<" end of test\n";
	return 0;
}