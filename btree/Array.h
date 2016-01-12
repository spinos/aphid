#pragma once

#include "Entity.h"
#include "Sequence.h"
namespace sdb {
template<typename KeyType, typename ValueType>
class Array : public Sequence<KeyType>
{
public:
    Array(Entity * parent = NULL) : Sequence<KeyType>(parent) {}
	
	virtual ~Array() {}
    
    void insert(const KeyType & x, ValueType * v) {
		Pair<KeyType, Entity> * p = Sequence<KeyType>::insert(x);
		if(!p->index) p->index = new Single<ValueType>;
		Single<ValueType> * d = static_cast<Single<ValueType> *>(p->index);
		d->setData(v);
	}
	
	ValueType * value() const {
		Single<ValueType> * s = static_cast<Single<ValueType> *>(Sequence<KeyType>::currentIndex());
		return s->data();
	}
	
	const KeyType key() const {
		return Sequence<KeyType>::currentKey();
	}
	
	ValueType * find(const KeyType & k, MatchFunction::Condition mf = MatchFunction::mExact, KeyType * extraKey = NULL) const
	{			
		Pair<Entity *, Entity> g = Sequence<KeyType>::findEntity(k, mf, extraKey);

		if(!g.index) return NULL;
		
		Single<ValueType> * s = static_cast<Single<ValueType> *>(g.index);
		
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
