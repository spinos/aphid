/*
 *  Primitive.h
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
class Geometry;
class Primitive {
public:
	Primitive();

	void setGeometry(Geometry * data);
	Geometry *getGeometry() const;
	
	void setComponentIndex(const unsigned &idx);
	const unsigned & getComponentIndex() const;
	
private:
	Geometry *m_geometry;
	unsigned m_componentIndex;
};