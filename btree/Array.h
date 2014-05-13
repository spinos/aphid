#pragma once

#include "Entity.h"
#include "Sequence.h"
namespace sdb {
template<typename KeyType, typename ValueType>
class Array : public Sequence<KeyType>
{
public:
    Array(Entity * parent = NULL) : Sequence<KeyType>(parent) {}
    
    void insert(const KeyType & x, ValueType * v) {
		Pair<KeyType, Entity> * p = Sequence<KeyType>::insert(x);
		if(!p->index) p->index = new Single<ValueType>;
		Single<ValueType> * d = static_cast<Single<ValueType> *>(p->index);
		d->setData(v);
	}
	
	ValueType * value() {
		Single<ValueType> * s = static_cast<Single<ValueType> *>(Sequence<KeyType>::currentIndex());
		return s->data();
	}
	
	const KeyType key() const {
		return Sequence<KeyType>::currentKey();
	}
	
	void remove(const KeyType & k) {
		Sequence<KeyType>::remove(k);
	}
};
} //end namespace sdb
