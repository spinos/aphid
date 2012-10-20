/*
 *  IndexArray.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <vector>

class IndexArray {
public:
	IndexArray();
	virtual ~IndexArray();
	
	void clear();
	
	unsigned * allocate(unsigned size);
	void push_back(const unsigned &value);
	
	void start();
	void take(const unsigned &value);
	void resizeToTaken();
	
	unsigned &operator[](unsigned index);
	const unsigned &operator[](unsigned index) const;
		
	unsigned size() const;
	unsigned blockCount() const;
	unsigned capacity() const;
	unsigned taken() const;
	
	static unsigned BlockSize;
	
private:
	std::vector<unsigned *> m_blocks;
	unsigned m_pos;
	unsigned m_current;
};