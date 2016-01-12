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
#include <Vector3F.h>
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
	const std::string str() const;
	friend std::ostream& operator<<(std::ostream &output, const Coord3 & p)
    {
        output << p.str();
        return output;
    }
	int x, y, z;
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