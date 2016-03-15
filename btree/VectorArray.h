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

namespace aphid {

namespace sdb {

template <typename T>
class VectorArray : public Entity {
	
	std::vector<T *> m_blocks;
	std::vector<T *> m_data;
	T * m_buf;
	int m_numData;
	
public:
	VectorArray(Entity * parent = NULL) : Entity(parent),
	m_numData(0) {}
	
	virtual ~VectorArray() 
	{ clear(); }
	
	void clear()
	{
		m_numData = 0;
		typename std::vector<T *>::iterator it = m_blocks.begin();
		for(;it!=m_blocks.end();++it) delete[] *it;
		m_blocks.clear();
		m_data.clear();
	}
	
	void insert()
	{
		if((m_numData & 1023)==0) {
			m_buf = new T[1024];
			m_blocks.push_back(m_buf);
		}
		
		T * d = &m_buf[m_numData & 1023];
		m_data.push_back(d);
		m_numData++;
	}
	
	void insert(const T & a) 
	{
		if((m_numData & 1023)==0) {
			m_buf = new T[1024];
			m_blocks.push_back(m_buf);
		}
		
		T * d = &m_buf[m_numData & 1023];
		*d = a;
		m_data.push_back(d);
		m_numData++;
	}
	
	T * get(int idx) const
	{ return m_data[idx]; }
	
	T * operator[] (int idx)
	{ return m_data[idx]; }
	
	const T * operator[] (int idx) const
	{ return m_data[idx]; }
	
	T * last ()
	{ return m_data[m_numData-1]; }
	
	const int & size() const;
	int numBlocks() const;
	T * block(const int & idx) const;
	
	int sizeInBytes() const;
	int numElementsInBlock(const int & idx) const;
	int elementBytes() const;
	
};

template <typename T>
const int & VectorArray<T>::size() const
{ return m_numData; }

template <typename T>
int VectorArray<T>::numBlocks() const
{ return m_blocks.size(); }

template <typename T>
T * VectorArray<T>::block(const int & idx) const
{ return m_blocks[idx]; }

template <typename T>
int VectorArray<T>::sizeInBytes() const
{ return sizeof(T) * m_numData; }

template <typename T>
int VectorArray<T>::numElementsInBlock(const int & idx) const
{
	if(idx < numBlocks() - 1) return 1024;
	return m_numData & 1023;
}

template <typename T>
int VectorArray<T>::elementBytes() const
{ return sizeof(T); }

}

}