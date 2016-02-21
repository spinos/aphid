/*
 *  VectorArray.h
 *  
 *
 *  Created by jian zhang on 1/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  un-organized array of *
 */
#pragma once
#include <vector>
#include "Entity.h"
namespace sdb {

template <typename T>
class VectorArray : public Entity {
	
	std::vector<T *> m_blocks;
	std::vector<T *> m_data;
	T * m_buf;
	unsigned m_numData;
	
public:
	VectorArray(Entity * parent = NULL) : Entity(parent),
	m_numData(0) {}
	
	virtual ~VectorArray() 
	{
		const int n = m_blocks.size();
		int i = 0;
		for(; i<n; i++) delete[] m_blocks[i];
		m_blocks.clear();
		m_data.clear();
	}
	
	void insert(const T & a) 
	{
		if((m_numData & 4095)==0) {
			m_buf = new T[4097];
			m_blocks.push_back(m_buf);
		}
		
		T * d = &m_buf[m_numData & 4095];
		*d = a;
		m_data.push_back(d);
		m_numData++;
	}
	
	const unsigned & size() const
	{ return m_numData; }
	
	T * get(int idx) const
	{ return m_data[idx]; }
};

}