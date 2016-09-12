/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 9/13/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *  http://eeweb.poly.edu/iselesni/WaveletSoftware/standard1D.html
 *  http://eeweb.poly.edu/iselesni/WaveletSoftware/dt2D.html
 *  http://fourier.eng.hmc.edu/e161/lectures/wavelets/node8.html
 *  http://eeweb.poly.edu/iselesni/DoubleSoftware/ddintro.html
 *  http://i-rep.emu.edu.tr:8080/xmlui/handle/11129/57
 *  http://cn.mathworks.com/help/wavelet/ref/dtfilters.html
 *  http://ocw.mit.edu/courses/mathematics/18-327-wavelets-filter-banks-and-applications-spring-2003/lecture-notes/
 *  http://www.stat.washington.edu/courses/stat530/spring08/PDFFILES/ACHA-Kingsbury.pdf
 *  http://www-sigproc.eng.cam.ac.uk/foswiki/pub/Main/NGK/ngk_ACHApap.pdf
 */

#include <iostream>
#include <map>
#include <boost/format.hpp>
#include <boost/timer.hpp>

/// x[N] N is 2x filter length M
/// af[2][M] 
/// 0 is lowpass filter, 1 is highpass filter
void afb(float * x, float **af)
{

}

/// w[j][k][d] DWT coefficients
/// j = 0...J-1, k = 0...1, d = 0...2
/// x M by N array
/// J number of stages
/// Faf first stage filters
/// af filters for remaining stages
///
void dualtree2D(float w[][][], float *x, int J, float *Faf, float *af)
{

}

int main(int argc, char * const argv[])
{
	std::cout<<"wavelet test\n";
	
	std::cout<<" end of test\n";
	return 0;
}
