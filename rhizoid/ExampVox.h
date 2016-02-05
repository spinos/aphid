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
	float m_diffuseMaterialColV[3];
	float * m_boxCenterSizeF4;
	unsigned m_numBoxes;
	
public:
	ExampVox();
	virtual ~ExampVox();
	
	virtual void voxelize(KdIntersection * tree);
	
protected:
	const BoundingBox & geomBox() const;
	void setGeomBox(const BoundingBox & x);
	void drawGrid();
	void drawWireGrid();
	float * diffuseMaterialColV();
	float * boxCenterSizeF4();
	const unsigned & numBoxes() const;
	void setNumBoxes(unsigned n);
	
private:

};