/*
 *  Ordered.h
 *  btree
 *
 *  Created by jian zhang on 5/9/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Sequence.h>
#include <List.h>
namespace aphid {
namespace sdb {

template<typename KeyType, typename ValueType>
class Ordered : public Sequence<KeyType>
{
public:
	Ordered(Entity * parent = NULL) : Sequence<KeyType>(parent) {}
	
	void insert(const KeyType & x, const ValueType & v) {
		Pair<KeyType, Entity> * p = Sequence<KeyType>::insert(x);
		if(!p->index) p->index = new List<ValueType>;
		static_cast<List<ValueType> *>(p->index)->insert(v);
	}
	
	List<ValueType> * value() {
		return static_cast<List<ValueType> *>(Sequence<KeyType>::currentIndex());
	}
	
	const KeyType key() const {
		return Sequence<KeyType>::currentKey();
	}
	
	int numElements() {
		int r = 0;
		Sequence<KeyType>::begin();
		while(!Sequence<KeyType>::end()) {
			r += value()->size();
			Sequence<KeyType>::next();
		}
		return r;
	}
	
	void elementBegin() {
		Sequence<KeyType>::begin();
		if(Sequence<KeyType>::end()) return;
		m_elemI = 0;
		m_elemSize = value()->size();
	}
	
	void nextElement() {
		m_elemI++;
		if(m_elemI == m_elemSize) {
			Sequence<KeyType>::next();
			if(Sequence<KeyType>::end()) return;
			m_elemI = 0;
			m_elemSize = value()->size();
		}
	}
	
	const bool elementEnd() const {
		return Sequence<KeyType>::end();
	}
	
	ValueType * currentElement() {
		return value()->valueP(m_elemI);
	}
	
	void remove(const KeyType & k, const ValueType & v) {
		Pair<Entity *, Entity> e = findEntity(k);
		if(!e.index) return;
		List<ValueType> * l = static_cast<List<ValueType> *>(e.index);
		l->remove(v);
		if(l->size() < 1) Sequence<KeyType>::remove(k);
	}
	
	void removeEmpty() {
		std::deque<KeyType> ks = Sequence<KeyType>::allKeys();
		typename std::deque<KeyType>::iterator it;
		it = ks.begin();
		for(; it != ks.end(); ++it) {
			Pair<Entity *, Entity> e = findEntity(*it);
			List<ValueType> *l = static_cast<List<ValueType> *>(e.index);
			if(l->size() < 1) Sequence<KeyType>::remove(*it);
		}
	}
	
private:
	int m_elemI, m_elemSize;
};
} //end namespace sdb
}
