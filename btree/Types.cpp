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

const std::string Coord2::str() const 
{
	return (boost::format("(%1%,%2%)") % x % y).str();
}

Coord3::Coord3() {x = y = z = 0;}
Coord3::Coord3(int a, int b, int c) { x =a; y= b; z=c; }

const bool Coord3::operator==(const Coord3 & another) const {
	return x == another.x && y == another.y && z == another.z;
}

const bool Coord3::operator<(const Coord3 & another) const {
	return (z< another.z &&	y< another.y && x< another.x);
}

const bool Coord3::operator>=(const Coord3 & another) const {
	return (z>= another.z && y>= another.y && x>= another.x);
}

const bool Coord3::operator>(const Coord3 & another) const {
	return (z> another.z &&	y> another.y && x> another.x);
}

const std::string Coord3::str() const 
{
	return (boost::format("(%1%,%2%,%3%)") % x % y % z).str();
}

const std::string V3::str() const 
{
	return (boost::format("(%1%,%2%,%3%)") % data[0] % data[1] % data[2]).str();
}

} // end namespace sdb