#include <iostream>
#include <QElapsedTimer>
#include <BaseArray.h>
#include <TypedEntity.h>
#include <Primitive.h>
#include <algorithm>

class Geom : public TypedEntity 
{
public:
	Geom() {printf("geom init ");}
};

class Mesh : public Geom 
{
public:
	Mesh() {printf("mesh init ");setMeshType();}
};

class Prim
{
public:
	Prim() {}
	unsigned geometry;
	unsigned component;
};

class A
{
public:
	A() {}
	void setA(int a) {m_a = a;}
	int getA() {return m_a;}
protected:	
	int m_a;
};

class B
{
public:
	static A *Ctx;
	B() {}
	int getCtxA() {return Ctx->getA();}
	int plus(int x) {Ctx->setA(getCtxA() + x); return Ctx->getA();}
};

A *B::Ctx = 0;

char * alignedPtr(unsigned size)
{
	char * p = new char[size + 31];
	return (char *)(((unsigned long)p + 32) & (0xffffffff - 31));
}

char isAlignedPtr(char* p)
{
	return ((unsigned)p & 31) == 0; 
}

const char *byte_to_binary(int x)
{
    static char b[33];
    b[32] = '\0';

    for (int z = 0; z < 32; z++) {
        b[31-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

    return b;

}

int main (int argc, char * const argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
	char *p = alignedPtr(512);
	char *p0 = alignedPtr(512);
	char *p1 = alignedPtr(512);
	printf("p  %s\n", byte_to_binary((unsigned)p));
	
	printf("p0 %s\n", byte_to_binary((unsigned)p0));
	printf("p1 %s\n", byte_to_binary((unsigned)p1));
	
	std::cout<<"p0 - p "<<p0 - p<<"\n";
	std::cout<<"p1 - p "<<p1 - p<<"\n";
	std::cout<<"p1 - p0 "<<p1 - p0<<"\n";
	
	if(isAlignedPtr(p)) std::cout<<"p is aligned\n";
	if(isAlignedPtr(p0)) std::cout<<"p0 is aligned\n";
	if(isAlignedPtr(p1)) std::cout<<"p1 is aligned\n";
	
	BaseArray a;
	a.setElementSize(4);
	a.expandBy(22);
	unsigned n = 75589;
	a.expandBy(n);
	
	a.verbose();
	
	QElapsedTimer timer;
	timer.start();
	for(unsigned i = 0; i < n; i++) {
		float *p = (float *)a.at(i);
		*p = 2.001f;
	}
	
	float sum = 0.f;
	
	//for(unsigned i = 0; i < n; i++) {
		for(unsigned j = 0; j < n; j++) {
			float *p = (float *)a.at(j);
			sum += *p;
			
		}
	//}
	
	printf("sum %f \n", sum);
	
	std::cout << "combine operation took " << timer.elapsed() << " milliseconds\n";
	
	//a.expand(302001);
	a.shrinkTo(22200);
	
	for(unsigned j = 0; j < 22200; j++) {
			float *p = (float *)a.at(j);
			*p = 0.f;
			
		}

	a.verbose();
	//a.expand(102001);
	//
	a.clear();
	
	std::vector<unsigned>primsmap;
	primsmap.push_back(32); // 0 - 31
	primsmap.push_back(100+32); // 32 - 131
	primsmap.push_back(41+100+32); // 132 - 172
	
	unsigned idx = 132;
	std::vector<unsigned>::const_iterator it = std::lower_bound(
				primsmap.begin(), primsmap.end(), idx+1);		
	printf("prim %i is %i", idx, it - primsmap.begin());
	
	A aa;
	aa.setA(99);
	
	B::Ctx = &aa;
	
	B b;
	printf("b.getA() = %i\n", b.getCtxA());
	
	aa.setA(71);
	
	printf("b.getA() = %i\n", b.getCtxA());
	
	B b1;
	printf("b1.plus(1) = %i\n", b1.plus(1));
	
	aa.setA(32);
	printf("b1.plus(1) = %i\n", b1.plus(1));
	printf("b.plus(2) = %i\n", b.plus(2));
	
	printf("a.getA() = %i\n", aa.getA());
	
	printf("2^20 is %i\n", (unsigned)1<<20);
	printf("2^20 as bits %s\n", byte_to_binary(1<<20));
	
	Mesh m;
	if(m.isMesh()) printf("m is mesh!\n");
	
	printf("size of mesh %i\n", sizeof(Mesh));
	printf("size of prim %i\n", sizeof(Prim));
	printf("size of Prim %i\n", sizeof(Primitive));
	Prim prm;
	prm.geometry = (unsigned)&m;
	if(((const Mesh *)prm.geometry)->isMesh()) printf("geom of prm is mesh!\n");
	

	return 0;
}
