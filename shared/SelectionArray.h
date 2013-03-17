/*
 *  SelectionArray.h
 *  lapl
 *
 *  Created by jian zhang on 3/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <vector>
class Primitive;
class Geometry;
class SelectionArray {
public:
	SelectionArray();
	virtual ~SelectionArray();
	
	void reset();
	void add(Primitive * prim);
	
	unsigned numPrims() const;
	Primitive * getPrimitive(const unsigned & idx) const;
private:
	std::vector<Primitive *> m_prims;
};