#pragma once

#include "Entity.h"
#include "Sequence.h"
namespace aphid {
namespace sdb {
template<typename KeyType, typename ValueType>
class Array : public Sequence<KeyType>
{
public:
	typedef Single<ValueType> SingleType;
	
    Array(Entity * parent = NULL) : Sequence<KeyType>(parent) {}
	
	virtual ~Array() {}
    
    void insert(const KeyType & x, ValueType * v) {
		Pair<KeyType, Entity> * p = Sequence<KeyType>::insert(x);
		if(p == NULL) {
		    throw "array cannot insert";
			return;
		}
		
		if(p->index == NULL) {
			try {
				p->index = new SingleType();
			} catch (std::bad_alloc& ba) {
				std::cerr << " array insert caught bad_alloc: "<< ba.what();
				return;
			} catch(...) {
				throw " array insert caught alloc index";
				return;
			}
		}
		
		try {
			SingleType * d = dynamic_cast<SingleType *>(p->index);
			d->setData(v);
		} catch(...) {
			throw "array insert caught set data";
		}
	}
	
	ValueType * value() const {
		SingleType * s = static_cast<SingleType *>(Sequence<KeyType>::currentIndex());
		return s->data();
	}
	
	ValueType * find(const KeyType & k, MatchFunction::Condition mf = MatchFunction::mExact, KeyType * extraKey = NULL) const
	{			
		Pair<Entity *, Entity> g = Sequence<KeyType>::findEntity(k, mf, extraKey);

		if(!g.index) return NULL;
		
		SingleType * s = static_cast<SingleType *>(g.index);
		
		return s->data();
	}
	
	virtual void clear() 
	{
		Sequence<KeyType>::begin();
		while(!Sequence<KeyType>::end()) {
			ValueType * p = value();
			if(p) delete p;
			Sequence<KeyType>::next();
		}
		
		Sequence<KeyType>::clear();
	}

private:
	
};
} //end namespace sdb
}
