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
	void finishInsert();
	int elementSize();
/// override HBase
	virtual char save();
	virtual char load();

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
	if(!p->index) {
		p->index = new ChildType(coord3Str(x), this);
		static_cast<ChildType *>(p->index)->createStorage(fObjectId);
		static_cast<ChildType *>(p->index)->close();
	}
	static_cast<ChildType *>(p->index)->insert((char *)&v);
}

template<typename ChildType, typename ValueType>
void HWorldGrid<ChildType, ValueType>::finishInsert()
{
	WorldGrid<ChildType, ValueType>::begin();
	while(!WorldGrid<ChildType, ValueType>::end() ) {
		WorldGrid<ChildType, ValueType>::value()->finishInsert();
		WorldGrid<ChildType, ValueType>::next();
	}
}

template<typename ChildType, typename ValueType>
int HWorldGrid<ChildType, ValueType>::elementSize()
{
	int sum = 0;
	WorldGrid<ChildType, ValueType>::begin();
	while(!WorldGrid<ChildType, ValueType>::end() ) {
		ChildType * cell = WorldGrid<ChildType, ValueType>::value();
		cell->open(fObjectId);
		sum += cell->numColumns();
		cell->close();
		
		WorldGrid<ChildType, ValueType>::next();
	}
	return sum;
}

template<typename ChildType, typename ValueType>
std::string HWorldGrid<ChildType, ValueType>::coord3Str(const Coord3 & c) const
{ return boost::str(boost::format("%1%_%2%_%3%") % c.x % c.y % c.z ); }

template<typename ChildType, typename ValueType>
char HWorldGrid<ChildType, ValueType>::save()
{
	HOocArray<hdata::TInt, 3, 256> * cellCoords = new HOocArray<hdata::TInt, 3, 256>(".cells");

	if(hasNamedData(".cells") ) {
		cellCoords->openStorage(fObjectId);
		cellCoords->clear();
	}
	else {
		cellCoords->createStorage(fObjectId);
	}
	
	int n=0;
	WorldGrid<ChildType, ValueType>::begin();
	while(!WorldGrid<ChildType, ValueType>::end() ) {
		Coord3 c = WorldGrid<ChildType, ValueType>::key();
		cellCoords->insert((char *)&c );
		WorldGrid<ChildType, ValueType>::next();
		n++;
	}
	
	cellCoords->finishInsert();
	
	std::cout<<"\n HWorldGrid save n grid "<<n;
	return 1;
}

template<typename ChildType, typename ValueType>
char HWorldGrid<ChildType, ValueType>::load()
{
	return 1;
}

}
