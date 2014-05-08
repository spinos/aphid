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
namespace sdb {

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

typedef Pair<int, V3> VertexP;

} // end namespace sdb