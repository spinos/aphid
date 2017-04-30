/*
 *  IndexList.h
 *  kdtree
 *
 *  Created by jian zhang on 10/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class IndexList {
public:
	IndexList();
	~IndexList();
	
	void create(const unsigned &num);
	unsigned *ptr();
private:
	char *m_raw;
	unsigned *m_aligned;
};