/*
 *  Test.h
 *  
 *
 *  Created by jian zhang on 4/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <string>

namespace sdb {

class Key3I
{
public:
	Key3I() {x = y = z = 0;}
	Key3I(int a, int b, int c) { x =a; y= b; z=c;}
	const bool operator<(const Key3I & another) const {
		return x < another.x || y < another.y || z < another.z;
	}
	const bool operator>=(const Key3I & another) const {
		return x >= another.x && y >= another.y && z >= another.z;
	}
	const bool operator==(const Key3I & another) const {
		return x == another.x && y == another.y && z == another.z;
	}
	int x, y, z;
};

template <typename KeyType, typename IndexType> 
class Pair
{
public:
	Pair() {}
	KeyType key;
	IndexType * index;
};

template <typename KeyType, typename IndexType> 
class Holder 
{
public:
	Holder() {}
	
	Pair<KeyType, IndexType> * data(int x) const;
	
	template <typename IndexType1>
	void take(const Pair<KeyType, IndexType1> & in) const {}
private:
	Pair<KeyType, int> m_data[4];
};

template <typename KeyType, typename IndexType> 
Pair<KeyType, IndexType> * Holder<KeyType, IndexType>::data(int x) const
{
	return new Pair<KeyType, IndexType>;
}

} // end of namespace sdb