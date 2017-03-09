/*
 *  gensig.cpp
 *  wvlt
 *
 *  Created by jian zhang on 9/17/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "gensig.h"
#include <math/ANoise3.h>

using namespace aphid;

void gen1dsig(UniformPlot1D * line,
				int m)
{
	line->create(m);
	int i=0, j;
	for(;i<m;++i) {

		float slope = (float)(i - m/2) / (float)m;
		slope = Absolute<float>(slope);
		slope = 1.f - slope;
		line->y()[i] = RandomFn11() * 0.13f * slope
					+ (cos(.031f * 3.14f * i) + 0.5f * cos(.5+.043f * 3.14f * i) )
					* .571f * slope;

	}
}

void gen2dsig(UniformPlot2D * img,
				int m, int n, int p,
				float noi)
{
	img->create(m, n, p);
	int i, j, k;
	for(k=0;k<p;++k) {
		float * Xv = img->y(k);
		
		for(j=0;j<n;++j) {
			for(i=0;i<m;++i) {

				float r = RandomFn11() * noi;

				Xv[j*m+i] = .5 + r * .5f;
			}
		}
	}
}

void gen2dsigFrac(UniformPlot2D * img,
				int m, int n, int p,
				int m0, int n0, int m1, int n1)
{
	float d = .01f;
	
	int i, j, k;
	for(k=0;k<p;++k) {
		float * Xv = img->y(k);
		
		Vector3F o(0.3435f + 0.1 * k, 0.23656f + 0.1 * k, 0.41765f + 0.1 * k);
		
		for(j=n0;j<n1;++j) {
			for(i=m0;i<m1;++i) {

				Vector3F p(d*(i-31), d*(j-31), d*(k-31) );
				float r = ANoise3::Fbm((const float *)&p,
												(const float *)&o,
												1.f,
												6,
												1.79f,
												.79f);
				Clamp01(r);
				r *= .9f;

				Xv[j*m+i] = r;
			}
		}
	}
}