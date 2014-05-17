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
#include <vector>
namespace sdb {

template<typename T>
class List : public Entity {
public:
	List(Entity * parent = NULL) : Entity(parent) 
	{
		m_numOccupied = 0;
		m_firstBlock = new Block;
	}
	
	virtual ~List() { 
		clear(); 
		delete m_firstBlock; 
	}
	
	const int size() const { return m_numOccupied; }
	
	void begin() {
		m_currentBlock = m_firstBlock;
		m_currentIt = 0;
		m_currentValue = m_currentBlock->m_data;
	}
	
	bool end() {
		return m_currentIt == m_numOccupied;
	}
	
	void next() {
		m_currentIt++;
		m_currentValue++;
		if(m_currentIt == m_numOccupied) return;
		if(m_currentIt % 256 == 0) {
			m_currentBlock = m_currentBlock->next();
			m_currentValue = m_currentBlock->m_data;
		}
	}
	
	void insert(const T & x) {
		Block * cur = findBlock(m_numOccupied);
		const int inBlock = m_numOccupied % 256;
		cur->m_data[inBlock] = x;
		m_numOccupied++;
	}
	
	int find(const T & x) {
		if(size() < 1) return -1;
		begin();
		while(!end()) {
			if(value() == x)
				return m_currentIt;
				
			next();
		}
		return -1;
	}
	
	void remove(const T & x) {
		int found = find(x);
		if(found < 0) return;
		
		T * rmed = valueP();
		
		T b = last();
		
		m_numOccupied--;
		
		if(size() == 0) return;
		
		*rmed = b;
		
		if(m_numOccupied % 256 == 0) {
			Block * b = findBlock(m_numOccupied - 1);
			b->removeNext();
		}
	}
	
	const T value() const {
		return *m_currentValue;
	}
	
	T * valueP() {
		return m_currentValue;
	}
	
	T last() {
		return value(m_numOccupied - 1);
	}
	
	T value(const int & i) const { 
		Block * b = findBlock(i);
		return b->m_data[i % 256];
	}
	
	T * valueP(const int & i) { 
		Block * b = findBlock(i);
		return &b->m_data[i % 256]; 
	}
	
	void clear() 
	{ 
		m_numOccupied = 0;
		if(m_firstBlock->next()) m_firstBlock->removeNext();
	}
private:
	class Block {
	public:
		Block() {
			m_next = NULL;
		}
		
		~Block() {
			if(m_next) delete m_next;
		}
		
		Block * next() {
			return m_next;
		}
		
		Block * addNext() {
			m_next = new Block;
			return m_next;
		}
		
		void removeNext() {
			if(m_next) {
				delete m_next;
				m_next = NULL;
			}
		}

		Block * m_next;
		T m_data[256];
	};
	
	Block * findBlock(const int & x) const {
		const int n = x / 256;
		Block * b = m_firstBlock;		
		for(int i = 0; i < n; i++) {
			if(!b->next()) b = b->addNext();
			else b = b->next();
		}
		return b;
	}
private:
	Block * m_firstBlock;
	Block * m_currentBlock;
	T * m_currentValue;
	int m_currentIt;
	int m_numOccupied;
};
} //end namespace sdb