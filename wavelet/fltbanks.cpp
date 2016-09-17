/*
 *  fltbanks.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "fltbanks.h"

namespace aphid {

namespace wla {

/// http://eeweb.poly.edu/iselesni/WaveletSoftware/allcode/FSfarras.m
/// one set of filters for each tree
const float Dtf::FirstStageUpFarrasAnalysis[2][10] = {
{0,
  -0.08838834764832,
   0.08838834764832,  
   0.69587998903400,  
   0.69587998903400,  
   0.08838834764832,  
  -0.08838834764832,   
   0.01122679215254, 
   0.01122679215254,  
                  0},
{  0,
  -0.01122679215254,
   0.01122679215254,
   0.08838834764832,
   0.08838834764832,
  -0.69587998903400,
   0.69587998903400,
  -0.08838834764832,
  -0.08838834764832,
   0}
};

const float Dtf::FirstStageUpFarrasSynthesis[2][10] = {
{0.00000000000000,
 0.01122679215254,
 0.01122679215254,
-0.08838834764832,
 0.08838834764832,
 0.69587998903400,
 0.69587998903400,
 0.08838834764832,
-0.08838834764832,
 0.00000000000000},
{0.00000000000000,
-0.08838834764832,
-0.08838834764832,
 0.69587998903400,
-0.69587998903400,
 0.08838834764832,
 0.08838834764832,
 0.01122679215254,
-0.01122679215254,
 0.00000000000000}
};

const float Dtf::FirstStageDownFarrasAnalysis[2][10] = {
{  0.01122679215254,                 
   0.01122679215254,                  
  -0.08838834764832,  
   0.08838834764832,  
   0.69587998903400,   
   0.69587998903400,  
   0.08838834764832,   
  -0.08838834764832,   
                  0,  
                  0}, 
{0,
 0,
-0.08838834764832,
-0.08838834764832,
 0.69587998903400,
-0.69587998903400,
 0.08838834764832,
 0.08838834764832,
 0.01122679215254,
-0.01122679215254}
};

const float Dtf::FirstStageDownFarrasSynthesis[2][10] = {
{0.00000000000000,
 0.00000000000000,
-0.08838834764832,
 0.08838834764832,
 0.69587998903400,
 0.69587998903400,
 0.08838834764832,
-0.08838834764832,
 0.01122679215254,
 0.01122679215254}, 
{
-0.01122679215254,
 0.01122679215254,
 0.08838834764832,
 0.08838834764832,
-0.69587998903400,
 0.69587998903400,
-0.08838834764832,
-0.08838834764832,
 0.00000000000000,
 0.00000000000000}
};

/// http://eeweb.poly.edu/iselesni/WaveletSoftware/allcode/dualfilt1.m
/// dual filters for stage > 1
const float Dtf::UpAnalysis[2][10] = {
{0.03516384000000, 
                  0,                  
  -0.08832942000000,  
   0.23389032000000,                  
   0.76027237000000,   
   0.58751830000000,  
                  0,   
  -0.11430184000000,   
                  0,                  
                  0},
{0,
 0,
-0.11430184000000,
 0,
 0.58751830000000,
-0.76027237000000,
 0.23389032000000,
 0.08832942000000,
 0,
-0.03516384000000
}
};

const float Dtf::DownAnalysis[2][10] = {
{0,
 0,                  
-0.11430184000000,   
 0,  
 0.58751830000000,  
 0.76027237000000,   
 0.23389032000000,                  
-0.08832942000000, 
 0,                 
 0.03516384000000},
{
-0.03516384000000,
 0,
 0.08832942000000,
 0.23389032000000,
-0.76027237000000,
 0.58751830000000,
 0,
-0.11430184000000,
 0,
 0}
};

const float Dtf::UpSynthesis[2][10] = {
{0.00000000,
 0.00000000,
-0.11430184,
 0.00000000,
 0.58751830,
 0.76027237,
 0.23389032,
-0.08832942,
 0.00000000,
 0.03516384},
{
-0.03516384,
 0.00000000,
 0.08832942,
 0.23389032,
-0.76027237,
 0.58751830,
 0.00000000,
-0.11430184,
 0.00000000,
 0.00000000
}
};

const float Dtf::DownSynthesis[2][10] = {
{0.03516384,
 0.00000000,
-0.08832942,
 0.23389032,
 0.76027237,
 0.58751830,
 0.00000000,
-0.11430184,
 0.00000000,
 0.00000000},
{0.00000000,
 0.00000000,
-0.11430184,
 0.00000000,
 0.58751830,
-0.76027237,
 0.23389032,
 0.08832942,
 0.00000000,
-0.03516384}
};

/// http://eeweb.poly.edu/iselesni/WaveletSoftware/allcode/farras.m
/// Farras nearly symmetric filters for orthogonal
/// 2-channel perfect reconstruction filter bank
const float Dtf::FarrasAnalysisFilter[2][10] = {
{0,
0,
-0.0883883476483,
0.0883883476483,
0.695879989034,
0.695879989034,
0.0883883476483,
-0.0883883476483,
0.0112267921525,
0.0112267921525},
{-0.0112267921525,
0.0112267921525,
0.0883883476483,
0.0883883476483,
-0.695879989034,
0.695879989034,
-0.0883883476483,
-0.0883883476483,
0,
0}
};

/// http://www.vincentcheung.ca/research/matlabindexrepmat.html
/// synthesis filter is row reverse of analysis filters
/// sf = af(end:-1:1, :)
/// all the rows and first column of x
/// x(:, 1)
/// all the rows and second column of x
/// x(:, 2)
const float Dtf::FarrasSynthesisFilter[2][10] = {
{0.0112267921525,
0.0112267921525,
-0.0883883476483,
0.0883883476483,
0.695879989034,
0.695879989034,
0.0883883476483,
-0.0883883476483,
0,
0},
{0,
0,
-0.0883883476483,
-0.0883883476483,
0.695879989034,
-0.695879989034,
0.0883883476483,
0.0883883476483,
0.0112267921525,
-0.0112267921525}
};

void Dtf::fsfarras(float ** af, float ** sf)
{
	for(int i=0; i<10; ++i) {
		af[0][i] = Dtf::FirstStageUpFarrasAnalysis[0][i];
		af[1][i] = Dtf::FirstStageUpFarrasAnalysis[1][i];
		af[2][i] = Dtf::FirstStageDownFarrasAnalysis[0][i];
		af[3][i] = Dtf::FirstStageDownFarrasAnalysis[1][i];
		sf[0][i] = Dtf::FirstStageUpFarrasSynthesis[0][i];
		sf[1][i] = Dtf::FirstStageUpFarrasSynthesis[1][i];
		sf[2][i] = Dtf::FirstStageDownFarrasSynthesis[0][i];
		sf[3][i] = Dtf::FirstStageDownFarrasSynthesis[1][i];
	}
}

void Dtf::dualflt(float ** af, float ** sf)
{
	for(int i=0; i<10; ++i) {
		af[0][i] = Dtf::UpAnalysis[0][i];
		af[1][i] = Dtf::UpAnalysis[1][i];
		af[2][i] = Dtf::DownAnalysis[0][i];
		af[3][i] = Dtf::DownAnalysis[1][i];
		sf[0][i] = Dtf::UpSynthesis[0][i];
		sf[1][i] = Dtf::UpSynthesis[1][i];
		sf[2][i] = Dtf::DownSynthesis[0][i];
		sf[3][i] = Dtf::DownSynthesis[1][i];
	}
}

void farras(float ** af, float ** sf)
{
	for(int i=0; i<10; ++i) {
		af[0][i] = Dtf::FarrasAnalysisFilter[0][i];
		af[1][i] = Dtf::FarrasAnalysisFilter[1][i];
		sf[0][i] = Dtf::FarrasSynthesisFilter[0][i];
		sf[1][i] = Dtf::FarrasSynthesisFilter[1][i];
	}
}

float * upsample(int & ny, const float * x, const int & n, 
				const int & p, const int & phase)
{
	ny = n * p;
	float * y = new float[ny];
	int i=0, j;
	for(;i<n;++i) {
		for(j=0;j<p;++j) {
			if(j==phase) 
				y[i*p+j] = x[i];
			else
				y[i*p+j] = 0.f;
		}
	}
	
	return y;
}

float * downsample(int & ny, const float * x, const int & n, 
				const int & p, const int & phase)
{
	int i=0, j=0;
	for(;i<n;++i) {
		if(i== j*p + phase) {
			j++;
		}
	}
	
	ny = j;
	
	float * y = new float[ny];
	
	i=j=0;
	for(;i<n;++i) {
		if(i== j*p + phase) {
			y[j++]=x[i];
		}
	}
	
	return y;
}

int periodic(const int & i, const int & n)
{
	if(i<0)
		return n+i;
	if(i>n-1)
		return i-n;
	return i;
}

/// http://cn.mathworks.com/help/matlab/ref/circshift.html
float * circshift(const float * x, const int & n, const int & p)
{
	float * y = new float[n];
	int i=0;
	for(;i<n;++i) {
		y[i] = x[periodic(i-p, n)];
	}
	return y;
}

/// http://learn.mikroe.com/ebooks/digitalfilterdesign/chapter/window-functions/
/// 1D impulse response from binomial weights
/// [1, 4, 6, 4, 1]/16
/// 2D approximation
/// [1  4  7  4 1]
/// [4 16 26 16 4]
/// [7 26 41 26 7]
/// [4 16 26 16 4]
/// [1  4  7  4 1]/273
/// finite impulse response express each output sample as a weighted sum of the last N input samples
/// y[n] = b0x[n] + b1x[n-1] + ... + bNx[n-N]
///      = sigma (i=0,N) bi x[n-i]
/// y[n] input signal
/// x[n] output signal
/// N filter order

float * fir(const float * x, const int & n,
			const float * w, const int & m)
{
	float * y = new float[n];
	
	int i=0, j;
	for(;i<n;++i) {
		y[i] = 0.f;
		
		for(j=0; j<m;++j) {
			y[i] += w[j] * x[periodic(i-j, n)];
		}
	}
	
	return y;
}

static const float HANNWL10[11] = {
-.045016, 0., .075026, .159155, .225079,
.25,
.225079, .159155, .075026, 0., -.045016};

static const float HANNWH10[11] = {
.045016f, 0.f, -.075026f, -.159155f, -.225079f,
.75f,
-.225079f, -.159155f, -.075026f, 0.f, .045016f};

float * hannLowPass(int & n)
{
	float * y = new float[11];
	int i=0;
	for(;i<11;++i)
		y[i] = HANNWL10[i];
		
	n = 11;
		
	return y;
}

float * hannHighPass(int & n)
{
	float * y = new float[11];
	int i=0;
	for(;i<11;++i)
		y[i] = HANNWH10[i];
		
	n = 11;
		
	return y;
}

float * firlHann(const float * x, const int & n)
{ return fir(x, n, HANNWL10, 11); }

float * firhHann(const float * x, const int & n)
{ return fir(x, n, HANNWH10, 11); }

void afb(const float * x, const int & n,
		float * & lo, float * & hi, int & m)
{
	float * lo2 = fir(x, n, Dtf::FarrasAnalysisFilter[0], 10);
	float * hi2 = fir(x, n, Dtf::FarrasAnalysisFilter[1], 10);
	lo = downsample(m, lo2, n, 2);
	hi = downsample(m, hi2, n, 2);
	
	delete[] lo2;
	delete[] hi2;
}

void sfb(const float * lo, const float * hi, const int & m,
		float * & y, int & n)
{
	float * lo2 = upsample(n, lo, m, 2);
	float * hi2 = upsample(n, hi, m, 2);
	float * lo2ft = fir(lo2, n, Dtf::FarrasSynthesisFilter[0], 10);
	float * hi2ft = fir(hi2, n, Dtf::FarrasSynthesisFilter[1], 10);
	
	float * yd = new float[n];
	for(int i=0; i<n;++i)
		yd[i] = lo2ft[i] + hi2ft[i];
		
/// 1 - 10/2
	y = circshift(yd, n, -4);
	
	delete[] lo2;
	delete[] hi2;
	delete[] lo2ft;
	delete[] hi2ft;
	delete[] yd;
}

void afbflt(const float * x, const int & n,
		VectorN<float> & lo, VectorN<float> & hi,
		const float flt[2][10])
{
	float * lo2 = fir(x, n, flt[0], 10);
	float * hi2 = fir(x, n, flt[1], 10);
	lo.downsample(lo2, n, 2);
	hi.downsample(hi2, n, 2);
	
	delete[] lo2;
	delete[] hi2;
}

void sfbflt(VectorN<float> & y,
		const VectorN<float> & lo, const VectorN<float> & hi,
		const float flt[2][10])
{
	const int m = lo.N();
	int n;
	float * lo2 = upsample(n, lo.v(), m, 2);
	float * hi2 = upsample(n, hi.v(), m, 2);
	float * lo2ft = fir(lo2, n, flt[0], 10);
	float * hi2ft = fir(hi2, n, flt[1], 10);
	
	y.create(n);
	for(int i=0; i<n;++i)
		y[i] = lo2ft[i] + hi2ft[i];
		
/// 1 - 10/2
	y.circshift(-4);
	
	delete[] lo2;
	delete[] hi2;
	delete[] lo2ft;
	delete[] hi2ft;
}
		
}

}