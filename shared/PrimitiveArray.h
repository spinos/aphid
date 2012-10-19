/*
 *  PrimitiveArray.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <vector>
#include <Primitive.h>

class PrimitiveArray {
public:
	PrimitiveArray();
	virtual ~PrimitiveArray();
	
	void clear();
	
	Primitive * allocate(unsigned size);
	void push_back(const Primitive &value);
	
	Primitive &operator[](unsigned index);
	const Primitive &operator[](unsigned index) const;
		
	unsigned size() const;
	unsigned blockCount() const;
	unsigned capacity() const;
	
	static unsigned BlockSize;
	
private:
	std::vector<Primitive *> m_blocks;
	unsigned m_pos;
};