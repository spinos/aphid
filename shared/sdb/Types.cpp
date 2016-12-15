/*
 *  Types.cpp
 *  btree
 *
 *  Created by jian zhang on 5/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Types.h"

#include <boost/format.hpp>
namespace aphid {

namespace sdb {

Coord2::Coord2() {x = y = 0;}
Coord2::Coord2(int a, int b) { x =a; y= b; }

const bool Coord2::operator==(const Coord2 & another) const {
	return x == another.x && y == another.y;
}

const bool Coord2::operator<(const Coord2 & another) const {
	if(y< another.y) return true;
	if(y> another.y) return false;
	
	if(x < another.x) return true;
	return false;
}

const bool Coord2::operator>=(const Coord2 & another) const {
	if(y> another.y) return true;
	if(y< another.y) return false;
	
	if(x >= another.x) return true;
	return false;
}

const bool Coord2::operator>(const Coord2 & another) const
{
	if(y> another.y) return true;
	if(y< another.y) return false;
	
	if(x > another.x) return true;
	return false;
}

Coord2 Coord2::ordered() const
{
	int a = x;
	if(y < a) a = y;
	
	int b = x;
	if(y > a) b = y;
	
	return Coord2(a, b);
}

const std::string Coord2::str() const 
{
	return (boost::format("(%1%,%2%)") % x % y).str();
}

Coord3::Coord3() 
{x = y = z = 0;}

Coord3::Coord3(int a, int b, int c) 
{ x =a; y= b; z=c; }

const bool Coord3::operator==(const Coord3 & another) const {
	return x == another.x && y == another.y && z == another.z;
}

const bool Coord3::operator<(const Coord3 & another) const {
	if(z < another.z) return true;
	if(z > another.z) return false;
/// z equals
	if(y < another.y) return true;
	if(y > another.y) return false;
/// y equals
	return x < another.x;
}

const bool Coord3::operator>=(const Coord3 & another) const {
	if(z > another.z) return true;
	if(z < another.z) return false;
/// z equals
	if(y > another.y) return true;
	if(y < another.y) return false;
/// y equals
	return x >= another.x;
}

const bool Coord3::operator>(const Coord3 & another) const {
	if(z > another.z) return true;
	if(z < another.z) return false;
/// z equals
	if(y > another.y) return true;
	if(y < another.y) return false;
/// y equals
	return x > another.x;
}

Coord3 Coord3::ordered() const
{
	int a = x;
	if(y < a) a = y;
	if(z < a) a = z;
	
	int c = x;
	if(y > c) c = y;
	if(z > c) c = z;
	
	int b = x;
	if(y > a && y < c) b = y;
	if(z > a && z < c) b = z;
	return Coord3(a, b, c);
}

void Coord3::makeUnique()
{
	if(z > y && z > x)
		return;
		
	const Coord3 d = *this;
	if(y > x && y > z) {
		x = d.z;
		y = d.x;
		z = d.y;
		return;
	}
	
	if(x > y && x > z) {
		x = d.y;
		y = d.z;
		z = d.x;
	}
}

const std::string Coord3::str() const 
{
	return (boost::format("(%1%,%2%,%3%)") % x % y % z).str();
}

bool Coord3::intersects(const int & v) const
{
	if(x == v) return true;
	if(y == v) return true;
	return (z == v);
}

bool Coord3::intersects(const Coord2 & e) const
{
	if(e.x == x && e.y == y ) return true;
	if(e.x == y && e.y == z ) return true;
	return (e.x == z && e.y == x );
}

const std::string V3::str() const 
{
	return (boost::format("(%1%,%2%,%3%)") % data[0] % data[1] % data[2]).str();
}

Coord4::Coord4() {x = y = z = w = 0;}
Coord4::Coord4(int a, int b, int c, int d) 
{ x =a; y= b; z=c; w = d; }

const bool Coord4::operator==(const Coord4 & another) const 
{
	return (x == another.x && y == another.y 
			&& z == another.z && w == another.w);
}

const bool Coord4::operator<(const Coord4 & another) const 
{
	if(w < another.w) return true;
	if(w > another.w) return false;
/// w equals	
	if(z < another.z) return true;
	if(z > another.z) return false;
/// z equals
	if(y < another.y) return true;
	if(y > another.y) return false;
/// y equals
	return x < another.x;
}

const bool Coord4::operator>=(const Coord4 & another) const 
{
	if(w > another.w) return true;
	if(w < another.w) return false;	
/// w equals
	if(z > another.z) return true;
	if(z < another.z) return false;
/// z equals
	if(y > another.y) return true;
	if(y < another.y) return false;
/// y equals
	return x >= another.x;
}

const bool Coord4::operator>(const Coord4 & another) const 
{
	if(w > another.w) return true;
	if(w < another.w) return false;
/// w equals
	if(z > another.z) return true;
	if(z < another.z) return false;
/// z equals
	if(y > another.y) return true;
	if(y < another.y) return false;
/// y equals
	return x > another.x;
}

const std::string Coord4::str() const 
{
	return (boost::format("(%1%,%2%,%3%,%4%)") % x % y % z % w).str();
}

} // end namespace sdb

}