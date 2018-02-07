/*
 *  SuperShape.h
 *  
 *  http://paulbourke.net/geometry/supershape/
 *
 *  Created by jian zhang on 11/30/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SUPER_SHAPE_H
#define APH_SUPER_SHAPE_H

#include <geom/GeodesicSphereMesh.h>
#include <boost/scoped_array.hpp>

namespace aphid {

struct SuperFormulaParam {

	float _a1, _b1, _m1, _n1, _n2, _n3, _a2, _b2, _m2, _n21, _n22, _n23;
	
	void reset()
	{
		_a1 = 1.f; _b1 = 1.f; _m1 = 0.f; _n1 = 1.f; _n2 = 1.f; _n3 = 1.f;
		_a2 = 3.f; _b2 = 3.f; _m2 = 1.f; _n21 = 1.f; _n22 = 1.f; _n23 = 1.f;
	}
};

class SuperShape {
	
	SuperFormulaParam m_param;
	
public:
	SuperShape();
	virtual ~SuperShape();
	
	SuperFormulaParam& param();
	
/// azimuth u [-pi,pi] v inclination [-0.5pi,0.5pi]
	void computePosition(float* dst,
		const float & u, const float & v) const;
	
protected:

private:
	
};

class SuperShapeGlyph : public SuperShape, public TriangleGeodesicSphere {
	
/// theta and phi
	boost::scoped_array<Float2> m_pcoord;
	
public:
	SuperShapeGlyph(int level = 31);
	virtual ~SuperShapeGlyph();
	
	void computePositions();
	
protected:

private:
	
};

}
#endif