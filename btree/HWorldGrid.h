/*
 *  HWorldGrid.h
 *  julia
 *
 *  Created by jian zhang on 1/3/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  out-of-core grid
 */

#pragma once

#include "WorldGrid.h"
#include <HBase.h>
#include <boost/format.hpp>

namespace sdb {

template<typename ChildType, typename ValueType>
class HWorldGrid : public HBase, public WorldGrid<ChildType, ValueType> {
	
public:
	HWorldGrid(const std::string & name, Entity * parent = NULL);
	virtual ~HWorldGrid();
	
	void insert(const float * at, const ValueType & v);

protected:
	std::string coord3Str(const Coord3 & c) const;
	
private:

};

template<typename ChildType, typename ValueType>
HWorldGrid<ChildType, ValueType>::HWorldGrid(const std::string & name, Entity * parent) :
HBase(name), WorldGrid<ChildType, ValueType>(parent)
{}

template<typename ChildType, typename ValueType>
HWorldGrid<ChildType, ValueType>::~HWorldGrid()
{}

template<typename ChildType, typename ValueType>
void HWorldGrid<ChildType, ValueType>::insert(const float * at, const ValueType & v) 
{
	const Coord3 x = WorldGrid<ChildType, ValueType>::gridCoord(at);
	
	Pair<Coord3, Entity> * p = Sequence<Coord3>::insert(x);
	if(!p->index) p->index = new ChildType(coord3Str(x), this);
	static_cast<ChildType *>(p->index)->insert(v);
}

template<typename ChildType, typename ValueType>
std::string HWorldGrid<ChildType, ValueType>::coord3Str(const Coord3 & c) const
{ return boost::str(boost::format("%1%_%2%_%3%") % c.x % c.y % c.z ); }

}
