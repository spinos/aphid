/*
 *  PixelProjector.cpp
 *  
 *
 *  Created by jian zhang on 2/5/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "PixelProjector.h"
#include "BufferBlock.h"
#include "ImageFragment.h"
#include <math/miscfuncs.h>

using namespace aphid;

PixelProjector::PixelProjector()
{}

PixelProjector::~PixelProjector()
{}

void PixelProjector::reproject(BufferBlock& blk, const ImageFragment& y_t)
{
	const int& t = blk.age();
	const float alpha = 1.f - 4.f / (4.f + t);
	float residual = 0.f;
	
	ImageFragment* h_tm1 = blk.fragment();
/// same size
	const int n = h_tm1->fragmentWidth() * h_tm1->fragmentHeight();
	
	float* redH = h_tm1->colorComponent(0); 
	const float* redY = y_t.colorComponent(0); 
	for(int i=0;i<n;++i) {
		Blend(redH[i], redY[i], alpha, residual);
	}
	
	float* greenH = h_tm1->colorComponent(1); 
	const float* greenY = y_t.colorComponent(1); 
	for(int i=0;i<n;++i) {
		Blend(greenH[i], greenY[i], alpha, residual);
	}
	
	float* blueH = h_tm1->colorComponent(2); 
	const float* blueY = y_t.colorComponent(2); 
	for(int i=0;i<n;++i) {
		Blend(blueH[i], blueY[i], alpha, residual);
	}
	
	float* alphaH = h_tm1->colorComponent(3); 
	const float* alphaY = y_t.colorComponent(3); 
	for(int i=0;i<n;++i) {
		Blend(alphaH[i], alphaY[i], alpha, residual);
	}
	
	residual /= (float)n;
	blk.setResidual(residual);
}

void PixelProjector::Overwrite(float& h_t, const float& y_t, float& residual)
{ 
	float err = h_t;
	
	h_t = y_t; 
	
	err -= h_t;
	residual += Absolute<float>(err);
}

void PixelProjector::Blend(float& h_t, const float& y_t, const float& a, 
								float& residual)
{
	float err = h_t;
	
	h_t *= a;
	h_t += (1.f - a) * y_t; 
	
	err -= h_t;
	residual += Absolute<float>(err);
}
