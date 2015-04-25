/*
 *  Geometry.h
 *  kdtree
 *
 *  Created by jian zhang on 10/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <TypedEntity.h>
#include <BoundingBox.h>
class Geometry : public TypedEntity {
public:
	Geometry();
	
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	
	void setBBox(const BoundingBox &bbox);
	const BoundingBox & getBBox() const;
	
protected:
	BoundingBox * bbox();
	void updateBBox(const BoundingBox & box);
	
private:
	BoundingBox m_bbox;
};