/*
 *  fltbanks.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef WLA_FLT_BANKS_H
#define WLA_FLT_BANKS_H
#include <ATypes.h>

namespace aphid {

namespace wla {

/// http://cn.mathworks.com/help/signal/ref/upsample.html
/// increase sampling rate by integer factor p with phase offset
/// by inserting p – 1 zeros between samples
float * upsample(int & ny, const float * x, const int & n, 
				const int & p, const int & phase = 0);
				
/// http://cn.mathworks.com/help/signal/ref/downsample.html
/// decrease sampling rate by integer factor p with phase offset
float * downsample(int & ny, const float * x, const int & n, 
				const int & p, const int & phase = 0);

int periodic(const int & i, const int & n);

/// delay (p>0) or haste (p<0) the signal
float * circshift(const float * x, const int & n, const int & p);

/// apply finite impulse response filter 
/// to signal X[N]
/// W[M] response coefficients
float * fir(const float * x, const int & n,
			const float * w, const int & m);

/// http://learn.mikroe.com/ebooks/digitalfilterdesign/chapter/examples/
/// Hann window low pass filter coefficients 11-tap
float * hannLowPass(int & n);
float * hannHighPass(int & n);

float * firlHann(const float * x, const int & n);
float * firhHann(const float * x, const int & n);

/// analysis and synthesis filters of order 10
/// af[2][10] analysis filter
/// sf[2][10] synthesis filter
/// first column is low pass, second is high pass
void farras(float ** af, float ** sf);

/// http://eeweb.poly.edu/iselesni/WaveletSoftware/allcode/afb.m
/// analysis filter bank
/// apply farras filters and downsample
/// X[N] input signal
/// lo low frequency output
/// hi high frequency output
/// M output length
void afb(const float * x, const int & n,
		float * & lo, float * & hi, int & m);

/// http://eeweb.poly.edu/iselesni/WaveletSoftware/allcode/sfb.m
/// synthesisi filter bank
/// upsample input, apply farras filters, add up
/// lo[M] low frequency input 
/// hi[M] high frequency input
/// M input length
/// Y output signal
/// N output length
void sfb(const float * lo, const float * hi, const int & m,
		float * & y, int & n);
	
/// get the filters
/// af[4][10] analysis filters
/// sf[4][10] synthesis filters
void fsfarras(float ** af, float ** sf);
void dualflt(float ** af, float ** sf);

/// dual tree first stage filters up and down
/// analysis
void afbDtFsU(const float * x, const int & n,
		VectorN<float> & lo, VectorN<float> & hi);
void afbDtFsD(const float * x, const int & n,
		VectorN<float> & lo, VectorN<float> & hi);
/// stage > 0
void afbDtU(const float * x, const int & n,
		VectorN<float> & lo, VectorN<float> & hi);
void afbDtD(const float * x, const int & n,
		VectorN<float> & lo, VectorN<float> & hi);
/// synthesis
void sfbDtFsU(VectorN<float> & y,
		const VectorN<float> & lo, const VectorN<float> & hi);
void sfbDtFsD(VectorN<float> & y,
		const VectorN<float> & lo, const VectorN<float> & hi);
/// stage > 0
void sfbDtU(VectorN<float> & y,
		const VectorN<float> & lo, const VectorN<float> & hi);
void sfbDtD(VectorN<float> & y,
		const VectorN<float> & lo, const VectorN<float> & hi);
}

}

#endif
