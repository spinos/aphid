/*
 *  GeodesicSphere.cpp
 *
 *  directional samples at each face center of a geodesic dome
 *  36 samples when level = 3
 *
 *  Created by jian zhang on 8/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GeodesicSphere.h"
#include <math/Vector3F.h>
#include <math/miscfuncs.h>

namespace aphid {

namespace smp {

GeodesicSphere::GeodesicSphere() :
m_samplePnts(NULL),
m_numSamples(0)
{}

GeodesicSphere::~GeodesicSphere()
{
	if(m_samplePnts)
		delete[] m_samplePnts;
}

static const int sGeodesicLevelNumSamples[6][2] = {{0,0},
{4,4},
{12,16},
{20,36},
{28,64},
{36,100}};

void GeodesicSphere::generateSamples(const float& angleLimit, int level)
{
	if(m_samplePnts)
		delete[] m_samplePnts;
		
	m_samplePnts = new Vector3F[sGeodesicLevelNumSamples[level][1]];
	
	const float db = angleLimit / (float)level;
	
	int acc=0;
	for(int j=1;j<=level;++j) {
		const float b = db * j;
		const float height = cos(b);
		const float radius = sin(b);
		const int& na = sGeodesicLevelNumSamples[j][0];
		const float da = TWOPIF / (float)na;
		for(int i=0;i<na;++i) {
			
			const float ang = da * i;
			Vector3F pa(sin(ang) * radius, height, cos(ang) * radius );
			pa.normalize();
			 
			m_samplePnts[acc++] = pa;
		}
	}
	
}

const int& GeodesicSphere::numSamples() const
{ return m_numSamples; }

const Vector3F& GeodesicSphere::getSample(int i) const
{ return m_samplePnts[i]; }

Vector3F GeodesicSphere::getSample(int i, float noi) const
{
	Vector3F res = getSample(i);
	res.x += RandomFn11() * noi;
	//res.y += RandomFn11() * noi; 
	res.z += RandomFn11() * noi;
	res.normalize();
	return res;
}

}
}
