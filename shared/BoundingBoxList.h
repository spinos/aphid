/*
 *  BoundingBoxList.h
 *  kdtree
 *
 *  Created by jian zhang on 10/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BoundingBox.h>
class BoundingBoxList {
public:
	BoundingBoxList();
	~BoundingBoxList();
	
	void create(const unsigned &num);
	BoundingBox *ptr();
private:
	char *m_raw;
	BoundingBox *m_aligned;
};