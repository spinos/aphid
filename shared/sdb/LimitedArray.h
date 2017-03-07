/*
 *  LimitedArray.h
 *  sdb
 *
 *  data in one limited trunk, no hierarchy
 *
 *  Created by jian zhang on 3/7/17.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_SDB_LIMITED_ARRAY_H
#define APH_SDB_LIMITED_ARRAY_H

#include "LimitedSequence.h"

namespace aphid {
namespace sdb {

template<typename KeyType, typename ValueType, int MaxNKey>
class LimitedArray : public LimitedSequence<KeyType, MaxNKey>
{
	typedef Single<ValueType> SingleTyp;
	typedef LimitedSequence<KeyType, MaxNKey> DerivedTyp;
	
public:
	
    LimitedArray(Entity * parent = NULL);
	virtual ~LimitedArray();
    
    void insert(const KeyType & x, ValueType * v);
	
	ValueType * find(const KeyType & k) const;
	
	ValueType * value() const;
	
	virtual void remove(const KeyType & x);
	virtual void clear();
/// keep data
	virtual void clearSequence();

private:
	
};

template<typename KeyType, typename ValueType, int MaxNKey>
LimitedArray<KeyType, ValueType, MaxNKey>::LimitedArray(Entity * parent) : DerivedTyp(parent) 
{}
	
template<typename KeyType, typename ValueType, int MaxNKey>
LimitedArray<KeyType, ValueType, MaxNKey>::~LimitedArray() 
{}

template<typename KeyType, typename ValueType, int MaxNKey>
void LimitedArray<KeyType, ValueType, MaxNKey>::insert(const KeyType & x, ValueType * v) 
{
	Pair<KeyType, Entity> * p = DerivedTyp::insert(x);
	if(p == NULL) {
		throw "LimitedArray cannot insert";
		return;
	}
	
	if(p->index == NULL) {
		try {
			p->index = new SingleTyp();
		} catch (std::bad_alloc& ba) {
			std::cerr << " LimitedArray insert caught bad_alloc: "<< ba.what();
			return;
		} catch(...) {
			throw " LimitedArray insert caught alloc index";
			return;
		}
	}
	
	try {
		SingleTyp * d = dynamic_cast<SingleTyp *>(p->index);
		d->setData(v);
	} catch(...) {
		throw "LimitedArray insert caught set data";
	}
}

template<typename KeyType, typename ValueType, int MaxNKey>
ValueType * LimitedArray<KeyType, ValueType, MaxNKey>::value() const 
{
	try {
	SingleTyp * s = dynamic_cast<SingleTyp *>(DerivedTyp::currentIndex());
	if(s==NULL) {
		throw "LimitedArray cast null single";
	}
	
	ValueType * r = s->data();
	if(r==NULL) {
		throw "LimitedArray value null";
	}
	return r;
	} catch(...) {
		throw "LimitedArray wrong value";
	}
}

template<typename KeyType, typename ValueType, int MaxNKey>
ValueType * LimitedArray<KeyType, ValueType, MaxNKey>::find(const KeyType & k) const
{			
	Pair<Entity *, Entity> g = DerivedTyp::findEntity(k);

	if(!g.index) {
		return NULL;
	}
	
	SingleTyp * s = dynamic_cast<SingleTyp *>(g.index);
	if(s == NULL) {
		throw "LimitedArray find null single";
	}
	
	return s->data();
}

template<typename KeyType, typename ValueType, int MaxNKey>
void LimitedArray<KeyType, ValueType, MaxNKey>::remove(const KeyType & x) 
{
	Pair<Entity *, Entity> g = DerivedTyp::findEntity(x);

	if(g.index) {
		SingleTyp * s = dynamic_cast<SingleTyp *>(g.index);
		if(s) {
			ValueType * r = s->data();
			if(r) {
				delete r;
			}
		}
	}
	
    DerivedTyp::remove(x);
}
	
template<typename KeyType, typename ValueType, int MaxNKey>
void LimitedArray<KeyType, ValueType, MaxNKey>::clear() 
{	
	DerivedTyp::begin();
	while(!DerivedTyp::end()) {
		ValueType * p = value();
		if(p) {
			delete p;
		}
		DerivedTyp::next();
	}
	
	DerivedTyp::clear();
}

template<typename KeyType, typename ValueType, int MaxNKey>
void LimitedArray<KeyType, ValueType, MaxNKey>::clearSequence() 
{
	DerivedTyp::clear();
}
	
} //end namespace sdb
}
#endif