/*
 *  Boundary.h
 *  aphid
 *
 *  Created by jian zhang on 4/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BoundingBox.h>
namespace aphid {

class Boundary {
	BoundingBox m_bbox;
	
public:
	Boundary();
	
	void setBBox(const BoundingBox &bbox);
	const BoundingBox & getBBox() const;
	
protected:
	void updateBBox(const BoundingBox & box);
	BoundingBox * bbox();
private:
	
};

}