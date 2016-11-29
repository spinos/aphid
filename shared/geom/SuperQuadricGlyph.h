/*
 *  SuperQuadricGlyph.h
 *  
 *
 *  Created by jian zhang on 11/30/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SUPER_QUADRIC_GLYPH_H
#define APH_SUPER_QUADRIC_GLYPH_H
#include <GeodesicSphereMesh.h>
#include <boost/scoped_array.hpp>

namespace aphid {

class SuperQuadricGlyph : public TriangleGeodesicSphere {
	
/// theta and phi
	boost::scoped_array<Float2> m_pcoord;
	
public:
	SuperQuadricGlyph(int level = 5);
	virtual ~SuperQuadricGlyph();
	
	void computePositions(const float & alpha, const float & beta);
	
protected:

private:
	void computePolarCoord();
	
};

}
#endif