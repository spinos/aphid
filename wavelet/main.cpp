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

/// http://cn.mathworks.com/help/signal/ref/upfirdn.html
/// upsampleing x by factor of 1 (no upsample)
/// filtering with impulse response sequence given in the vector (or matrix) first column of af
/// downsampling by factor of 2
/// lo = upfirdn(x, af(:,1), 1, 2);
/// http://cn.mathworks.com/help/matlab/ref/rand.html
/// 1-by-64 array of random numbers
/// x = rand(1,64);
/// http://cn.mathworks.com/help/matlab/ref/length.html
/// the length of the largest array dimension in X
/// L = length(af)/2;

/// analysis function
/// x[N] N is 2x filter length L
/// N numbers of x
/// af[L] 
/// 0 is lowpass filter, 1 is highpass filter
void afb(const float * x, const int & n, float ** af, const int & filterOrder=10)
{
/// width of filter
	int L = filterOrder/2;
	float * Xshf = wla::circshift(x, n, -L);
	
	float * Xflt = wla::fir(Xshf, n, af[0], filterOrder);
	delete[] Xshf;
	delete[] Xflt;
}

/// w[j][k][d] DWT coefficients
/// j = 0...J-1, k = 0...1, d = 0...2
/// x M by N array
/// J number of stages
/// Faf first stage filters
/// af filters for remaining stages
///
void dualtree2D(float **w, float *x, int J, float *Faf, float *af)
{

}

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

int main(int argc, char * const argv[])
{
	std::cout<<"wavelet test\n";
	
	testUpsampleCshift();
	
	std::cout<<" end of test\n";
	return 0;
}
