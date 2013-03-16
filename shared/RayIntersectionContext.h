/*
 *  RayIntersectionContext.h
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BoundingBox.h>

class RayIntersectionContext {
public:
	RayIntersectionContext();
	virtual ~RayIntersectionContext();
	
	void setBBox(const BoundingBox & bbox);
	BoundingBox getBBox() const;
	
	void verbose() const;

	BoundingBox m_bbox;
	int m_level;
	
private:
};