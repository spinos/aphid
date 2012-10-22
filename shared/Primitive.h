/*
 *  Primitive.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class Primitive {
public:
	Primitive();

	void setGeometry(char * data);
	char *getGeometry();
	
	void setComponentIndex(const unsigned &idx);
	const unsigned getComponentIndex() const;
	
	bool isMeshGeometry() const;
private:
	char *m_geometry;
	unsigned m_componentIndex;
};