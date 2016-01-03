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
	
	std::vector<T *> m_data;
	
public:
	VectorArray(Entity * parent = NULL) : Entity(parent) {}
	
	virtual ~VectorArray() 
	{
		const int n = m_data.size();
		int i = 0;
		for(; i<n; i++) delete m_data[i];
			
		m_data.clear();
	}
	
	void insert(const T & a) 
	{
		T * d = new T;
		*d = a;
		m_data.push_back(d); 
	}
	
	void insert(T * d) 
	{
		m_data.push_back(d); 
	}
	
	int size() const
	{ return m_data.size(); }
	
	T * get(int idx) const
	{ return m_data[idx]; }
};

}