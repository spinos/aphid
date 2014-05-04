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
#include <Entity.h>
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
class Pair1
{
public:
	Pair1() {}
	KeyType key;
	IndexType * index;
};

template <typename KeyType, typename IndexType> 
class Holder 
{
public:
	Holder() {}
	
	Pair1<KeyType, IndexType> * data(int x) const;
	
	template <typename IndexType1>
	void take(const Pair1<KeyType, IndexType1> & in) const {}
private:
	Pair1<KeyType, int> m_data[4];
};

template <typename KeyType, typename IndexType> 
Pair1<KeyType, IndexType> * Holder<KeyType, IndexType>::data(int x) const
{
	return new Pair1<KeyType, IndexType>;
}


class Base
{
public:
	void imp() {std::cout<<"imp base\n";}
};

class Leaf : public Base
{
public:
};

template <typename KeyType, typename IndexType> 
class Pair2 : public Base
{
public:
	KeyType key;
	IndexType * index;
};

template<typename K, typename I, typename L>
class A : public Base
{
public:
	void initChild() { m_child = new Pair2<K, I>[2];}
	void initChildC() { m_child = new Pair2<K, L>[2];}
	void imp() const { std::cout<<"imp a\n";}
	
	void pc() const {
		static_cast<Pair2<K, I> * >(m_child)->index->imp();
	}
	
	void pcc() const {
		static_cast<Pair2<K, L> * >(m_child)->index->imp();
	}
private:
	Base * m_child;
};

class NodeLeaf;

class NodeInterior : public A<int, NodeInterior, NodeLeaf>
{
public:
	void imp() const { std::cout<<"imp i\n";}
};

class NodeLeaf : public A<int, NodeLeaf, NodeLeaf>
{
public:
	void imp() const { std::cout<<"imp l\n";}
};

} // end of namespace sdb