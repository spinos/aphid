/*
 *  gensig.cpp
 *  wvlt
 *
 *  Created by jian zhang on 9/17/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "gensig.h"
#include <ANoise3.h>

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
				int m, int n, int p)
{
	img->create(m, n, p);
	
	float d = .012352f;
	
	int i, j, k;
	for(k=0;k<p;++k) {
		float * Xv = img->y(k);
		
		Vector3F o(0.6435f + 0.1 * k, 0.53656f + 0.1 * k, 0.71765f + 0.1 * k);
		
		for(j=0;j<n;++j) {
			for(i=0;i<m;++i) {

				Vector3F p(d*(i-31), d*(j-31), d*(k-31) );
				float r = ANoise3::Fbm((const float *)&p,
												(const float *)&o,
												3.23f,
												11,
												1.4141f,
												.853f);
				Clamp01(r);
				r *= .9f;

				Xv[j*m+i] = r;
			}
		}
	}
}