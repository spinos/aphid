/*
 *  SuperQuadricGlyph.cpp
 *  
 *
 *  Created by jian zhang on 11/30/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "SuperQuadricGlyph.h"

namespace aphid {

SuperQuadricGlyph::SuperQuadricGlyph(int level) : TriangleGeodesicSphere(level)
{
	m_pcoord.reset(new Float2[numPoints()]);
	computePolarCoord();
}

SuperQuadricGlyph::~SuperQuadricGlyph()
{}

void SuperQuadricGlyph::computePolarCoord()
{

	const Vector3F * src = vertexNormals();
	Float2 * dst = m_pcoord.get();
	const int n = numPoints();
	for(int i=0;i<n;++i) {
		const Vector3F & pcat = src[i];
		Float2 & ppl = dst[i];
		ppl.x = acos(pcat.z);
		
		if(pcat.x==0 ) {
			if( pcat.y >0) 
				ppl.y = HALFPIF;
			else 
				ppl.y = ONEHALFPIF;
			
		}
		else if(pcat.x>0 && pcat.y >=0) ppl.y = atan(pcat.y/pcat.x);
		else if(pcat.x<0 && pcat.y >=0) ppl.y = PIF - atan(-pcat.y/pcat.x);
		else if(pcat.x<0 && pcat.y <=0) ppl.y = PIF + atan(pcat.y/pcat.x);
		else ppl.y = TWOPIF + atan(pcat.y/pcat.x);
	}
}

void SuperQuadricGlyph::computePositions(const float & alpha, const float & beta)
{
	float stheta, ctheta, sphi, cphi;
	
	Vector3F * dst = points();
	const Float2 * src = m_pcoord.get();
	const int n = numPoints();
	for(int i=0;i<n;++i) {
		const Float2 & ppl = src[i];
		
		stheta = sin(ppl.x);
		ctheta = cos(ppl.x);
		sphi = sin(ppl.y);
		cphi = cos(ppl.y);
		
		dst[i].set(pow(Absolute<float>(stheta), alpha) * GetSign(stheta) * pow(Absolute<float>(cphi), beta) * GetSign(cphi),
					pow(Absolute<float>(stheta), alpha) * GetSign(stheta) * pow(Absolute<float>(sphi), beta) * GetSign(sphi),
					pow(Absolute<float>(ctheta), beta) * GetSign(ctheta) );

	}
}

}