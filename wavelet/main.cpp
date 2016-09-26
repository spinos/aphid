/*
 *  main.cpp
 *  
 *  Created by jian zhang on 9/13/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *  http://eeweb.poly.edu/iselesni/WaveletSoftware/standard1D.html
 *  x = rand(1,64);           
 *  [lo, hi] = afb(x, af);
 *  y = sfb(lo, hi, sf);
 *  http://eeweb.poly.edu/iselesni/WaveletSoftware/dt2D.html
 *  http://fourier.eng.hmc.edu/e161/lectures/wavelets/node8.html
 *  http://eeweb.poly.edu/iselesni/DoubleSoftware/ddintro.html
 *  http://i-rep.emu.edu.tr:8080/xmlui/handle/11129/57
 *  http://cn.mathworks.com/help/wavelet/ref/dtfilters.html
 *  http://ocw.mit.edu/courses/mathematics/18-327-wavelets-filter-banks-and-applications-spring-2003/lecture-notes/
 *  http://www.stat.washington.edu/courses/stat530/spring08/PDFFILES/ACHA-Kingsbury.pdf
 *  http://www-sigproc.eng.cam.ac.uk/foswiki/pub/Main/NGK/ngk_ACHApap.pdf
 *  http://www-sigproc.eng.cam.ac.uk/Main/NGK
 *  http://robotics.stanford.edu/%7Escohen/lect02/lect02.html
 */

#include <Calculus.h>
#include <map>
#include <boost/format.hpp>
#include <boost/timer.hpp>
#include "fltbanks.h"
#include "dwt2.h"

using namespace aphid;

void testUpsampleDownsample()
{
	std::cout<<"\n test upsample downsample";
	float X[16];
	int i=0;
	for(;i<16;++i)
		X[i] = i+1;
		
	calc::printValues<float>("x", 16, X);
	
	int nXup;
	float * Xup = wla::upsample(nXup, X, 16, 2);
	
	calc::printValues<float>("xup", nXup, Xup);
	
	int nXdn;
	float * Xdn = wla::downsample(nXdn, Xup, nXup, 2);
	
	calc::printValues<float>("xdn", nXdn, Xdn);
}

void testUpsampleCshift()
{
	std::cout<<"\n test circshift";
	float X[10];
	int i=0;
	for(;i<10;++i)
		X[i] = i+1;
		
#if 0
	float * Xshf = wla::circshift(X, 10, -3);
	calc::printValues<float>("Xshf", 10, Xshf);
#else
	wla::circleShift1(X, 10, 4);
	calc::printValues<float>("X", 10, X);
#endif
}

void testAnalysis()
{
	float X[64];
	int i=0;
	for(;i<64;++i)
		X[i] = RandomF01();
	calc::printValues<float>("X", 64, X);
	
	float * Xshf = wla::circshift(X, 64, -5);
		
	float * Xflt = wla::firlHann(Xshf, 64);
	calc::printValues<float>("Xflt", 64, Xflt);
}

void testCshift()
{
	VectorN<int> X;
	X.create(10);
	int i=0;
	for(;i<10;++i)
		X[i] = i+1;
		
	//X.circshift(1);
	VectorN<int> Y;
	Y.copy(X, -1);
	calc::printValues<int>("Cshift", 10, Y.v() );
}

void testSumDifference()
{
	Array2<float> a, b;
	a.create(4,3);
	b.create(4,3);
	float * va = a.v();
	float * vb = b.v();
	for(int i=0;i<12;++i) {
		va[i] = i+1;
		vb[i] = -i-1;
	}
	
	Array2<float> sum = a + b;
	sum *= 0.707106781187;
	std::cout<<"\n S "<<sum;
	
	Array2<float> diff = a - b;
	diff *= 0.707106781187;
	std::cout<<"\n D "<<diff;
	
	std::cout<<"\n a "<<a;
	std::cout<<"\n b "<<b;
	
	Array2<float> a1 = sum + diff;
	a1 *= 0.707106781187;
	std::cout<<"\n a1 "<<a1;
	
	Array2<float> b1 = sum - diff;
	b1 *= 0.707106781187;
	std::cout<<"\n b1 "<<b1;
}

int main(int argc, char * const argv[])
{
	std::cout<<"wavelet test\n";
	
	testSumDifference();
	
	std::cout<<" end of test\n";
	return 0;
}
