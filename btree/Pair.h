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
	Holder();
private:
	Pair<KeyType, IndexType> m_data[4];
};
} // end of namespace SD