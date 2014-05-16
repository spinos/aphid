/*
 *  List.h
 *  btree
 *
 *  Created by jian zhang on 5/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Entity.h>
#include <deque>
#include <vector>
namespace sdb {

template<typename T>
class List : public Entity {
public:
	List(Entity * parent = NULL) : Entity(parent) 
	{
		m_numOccupied = 0;
	}
	
	virtual ~List() { clear(); }
	
	int size() const { return m_numOccupied; }
	
	void insert(const T & x) {
		if(m_numOccupied == m_que.size())
			m_que.push_back(x);
		else
			m_que[m_numOccupied] = x;
			
		m_numOccupied++;
	}
	
	void remove(const T & x) {
		if(size() < 1) return;
		int i = 0;
		for(; i < size(); i++) {
			if(m_que[i] == x)
				break;
		}
		
		if(i == size()) return;
				
		m_numOccupied--;
		
		if(size() == 0) return;
		
		m_que[i] = m_que[size()];
	}
	
	const T value(const int & i) const { return m_que[i]; }
	T * valueP(const int & i) { return &m_que[i]; }
	
	void getValues(std::vector<T> & dst) const {
		typename std::deque<T>::const_iterator it;
		it = m_que.begin();
		for(; it != m_que.end(); ++it) dst.push_back(*it); 
	}
	
	void clear() 
	{ 
		m_numOccupied = 0;
		m_que.clear(); 
	}
private:
	
private:
	std::deque<T> m_que;
	int m_numOccupied;
};
} //end namespace sdb