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
	
	const List<ValueType> * value() const {
		return static_cast<List<ValueType> *>(Sequence<KeyType>::currentIndex());
	}
	
	const KeyType key() const {
		return Sequence<KeyType>::currentKey();
	}
	
private:
	
};
} //end namespace sdb
