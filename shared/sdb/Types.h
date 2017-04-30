/*
 *  Types.h
 *  btree
 *
 *  Created by jian zhang on 5/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <string>
#include <math/Vector3F.h>
namespace aphid {

namespace sdb {

class Coord2
{
public:
	Coord2();
	Coord2(int a, int b);
	const bool operator==(const Coord2 & another) const;
	const bool operator<(const Coord2 & another) const;
	const bool operator>=(const Coord2 & another) const;
	const bool operator>(const Coord2 & another) const;
	Coord2 ordered() const;
	const std::string str() const;
	friend std::ostream& operator<<(std::ostream &output, const Coord2 & p)
    {
        output << p.str();
        return output;
    }
	int x, y;
};

class Coord3
{
public:
	Coord3();
	Coord3(int a, int b, int c);
	const bool operator==(const Coord3 & another) const;
	const bool operator<(const Coord3 & another) const;
	const bool operator>=(const Coord3 & another) const;
	const bool operator>(const Coord3 & another) const;
    Coord3 operator+(const Coord3 & another) const;
	Coord3 ordered() const;
/// z as highest and keep in order
	void makeUnique();
	const std::string str() const;
	friend std::ostream& operator<<(std::ostream &output, const Coord3 & p)
    {
        output << p.str();
        return output;
    }
	
	bool intersects(const int & v) const;
	bool intersects(const Coord2 & e) const;
	
	int x, y, z;
};

/// w first
class Coord4
{
public:
	Coord4();
	Coord4(int a, int b, int c, int d);
	const bool operator==(const Coord4 & another) const;
	const bool operator<(const Coord4 & another) const;
	const bool operator>=(const Coord4 & another) const;
	const bool operator>(const Coord4 & another) const;
//	Coord4 ordered() const;
/// w as highest and keep in order
//	void makeUnique();
	const std::string str() const;
	friend std::ostream& operator<<(std::ostream &output, const Coord4 & p)
    {
        output << p.str();
        return output;
    }
	int x, y, z, w;
};

class V3 {
public:
	V3() { data[0] = data[1] = data[2] = 0.f; }
	V3(float *d) { set(d); }
	void set(float *d) {data[0] = d[0]; data[1] = d[1]; data[2] = d[2];}
	float data[3];
	
	const std::string str() const;
	friend std::ostream& operator<<(std::ostream &output, const V3 & p)
    {
        output << p.str();
        return output;
    }
};

template <typename KeyType, typename IndexType> 
class Pair
{
public:
	Pair() { index = NULL; }
	KeyType key;
	IndexType * index;
	friend std::ostream& operator<<(std::ostream &output, const Pair & p)
    {
        output << p.key << " : " << *p.index;
        return output;
    }
};

template<typename T1, typename T2>
class Couple
{
public:
	T1 * t1;
	T2 * t2;
	
	Couple() 
	{
		t1 = new T1;
		t2 = new T2;
	}
};

template<typename T1, typename T2, typename T3>
class Triple
{
public:
	T1 * t1;
	T2 * t2;
	T3 * t3;
	
	Triple()
	{
		t1 = new T1;
		t2 = new T2;
		t3 = new T3;
	}
};

template<typename T1, typename T2, typename T3, typename T4>
class Quadruple
{
public:
	T1 * t1;
	T2 * t2;
	T3 * t3;
	T4 * t4;
	
	Quadruple()
	{
		t1 = new T1;
		t2 = new T2;
		t3 = new T3;
		t4 = new T4;
	}
};

typedef Quadruple<Vector3F, Vector3F, Vector3F, float > PNPrefW;
class VertexP : public Pair<int, PNPrefW>
{
public:
	
	const bool operator==(const VertexP & another) const {
		return index == another.index;
	}
};

} // end namespace sdb

}