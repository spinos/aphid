/*
 *  ExampVox.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <DrawBox.h>
#include <BoundingBox.h>

class KdIntersection;
class ExampVox : public DrawBox {

	BoundingBox m_geomBox;
	Vector3F m_geomCenter;
	float * m_boxCenterSizeF4;
	float m_diffuseMaterialColV[3];
	float m_geomScale[3];
/// radius of bbox
	float m_geomExtent;
/// radius of x-z axis of bbox
	float m_geomSize;
	unsigned m_numBoxes;
	
public:
	ExampVox();
	virtual ~ExampVox();
	
	virtual void voxelize(KdIntersection * tree);
	
	void setGeomBox(const float & a, 
					const float & b,
					const float & c,
					const float & d,
					const float & e,
					const float & f);
	
	const float & geomExtent() const;
	const float & geomSize() const;
	const BoundingBox & geomBox() const;
	const float * geomCenterV() const;
	const Vector3F & geomCenter() const;
	const float * geomScale() const;
	const float * diffuseMaterialColor() const;
	
protected:
	void drawGrid();
	void drawWireGrid();
	float * diffuseMaterialColV();
	float * boxCenterSizeF4();
	bool setNumBoxes(unsigned n);
	const unsigned & numBoxes() const;
	
private:

};