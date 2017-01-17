/*
 *  ExampVox.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <ogl/DrawBox.h>
#include <ogl/DrawDop.h>
#include <ogl/DrawTriangle.h>
#include <math/BoundingBox.h>
#include <geom/Geometry.h>
#include <sdb/VectorArray.h>
#include <ConvexShape.h>
#include <math/AOrientedBox.h>

namespace aphid {

class ExampVox : public DrawBox, public DrawDop, public DrawTriangle {

	BoundingBox m_geomBox;
	Vector3F m_geomCenter;
	Vector3F m_dopSize;
	
	float m_diffuseMaterialColV[3];
/// radius of bbox
	float m_geomExtent;
/// radius of exclusion
	float m_geomSize;
/// scaling radius of exclusion
	float m_sizeMult;
	
public:
	ExampVox();
	virtual ~ExampVox();
	
	virtual void voxelize2(sdb::VectorArray<cvx::Triangle> * tri,
							const BoundingBox & bbox);
	
/// set b4 geom box
	void setGeomSizeMult(const float & x);
	void setGeomBox(BoundingBox * bx);
	void setDopSize(const float & a,
	                const float & b,
	                const float &c);
	
	const float & geomExtent() const;
	const float & geomSize() const;
	const BoundingBox & geomBox() const;
	const Vector3F & geomCenter() const;
	const float * diffuseMaterialColor() const;
	const float * dopSize() const;
	
	virtual void drawWiredBound() const;
	virtual void drawSolidBound() const;
	
protected:
	float * diffuseMaterialColV();
	void buildBounding8Dop(const BoundingBox & bbox);
						
private:
	
};

}