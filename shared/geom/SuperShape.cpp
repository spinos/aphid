/*
 *  SuperShape.cpp
 *  
 *
 *  Created by jian zhang on 11/30/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "SuperShape.h"
#include <math/miscfuncs.h>

namespace aphid {

SuperShape::SuperShape()
{
	m_param.reset();
}

SuperShape::~SuperShape()
{}

SuperFormulaParam& SuperShape::param()
{ return m_param; }

void SuperShape::computePosition(float* dst,
		const float & u, const float & v) const
{
	float raux1 = pow(Absolute<float>(1.f / m_param._a1 * Absolute<float>(cos(m_param._m1 * u / 4.f) ) ),
						m_param._n2)
				+ pow(Absolute<float>(1.f / m_param._b1 * Absolute<float>(sin(m_param._m1 * u / 4.f) ) ),
						m_param._n3);
	float r1 = pow(Absolute<float>(raux1), -1.f / m_param._n1 );
	float raux2 = pow(Absolute<float>(1.f / m_param._a2 * Absolute<float>(cos(m_param._m2 * v / 4.f) ) ),
						m_param._n22)
				+ pow(Absolute<float>(1.f / m_param._b2 * Absolute<float>(sin(m_param._m2 * v / 4.f) ) ),
						m_param._n23);
	float r2 = pow(Absolute<float>(raux2), -1.f / m_param._n21 );			
	dst[0] = r1 * cos(u) * r2 * cos(v);
	dst[1] = r1 * sin(u) * r2 * cos(v);
	dst[2] = r2 * sin(v);
}

SuperShapeGlyph::SuperShapeGlyph(int level) : TriangleGeodesicSphere(level)
{
	m_pcoord.reset(new Float2[numPoints()]);
	computeSphericalCoord(m_pcoord.get() );
	computePositions();
}

SuperShapeGlyph::~SuperShapeGlyph()
{}

void SuperShapeGlyph::computePositions()
{
	const Float2 * src = m_pcoord.get();
	const int n = numPoints();
	for(int i=0;i<n;++i) {
		const Float2 & ppl = src[i];
		float* dst = (float*)&points()[i];
		computePosition(dst, ppl.x, ppl.y);

	}
}

}