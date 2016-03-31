#ifndef HWORLDGRID_H
#define HWORLDGRID_H

/*
 *  HWorldGrid.h
 *  julia
 *
 *  Created by jian zhang on 1/3/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  out-of-core grid
 *  child must be ooc 
 */

#include "WorldGrid.h"
#include <HBase.h>
#include <HOocArray.h>
#include <boost/format.hpp>

namespace aphid {
    
namespace sdb {

class HVarGrid : public HBase {
    
    float m_bbx[6];
    int m_vtyp;
    float m_gsize;
    
public:
    HVarGrid(const std::string & name);
	virtual ~HVarGrid();
	
	virtual char verifyType();
	virtual char load();
	
	const int & valueType() const;
	const float & gridSize() const;
	const float * bbox() const;
};
    
template<typename ChildType, typename ValueType>
class HWorldGrid : public HBase, public WorldGrid<ChildType, ValueType> {

	std::string m_dirtyName;
	Sequence<Coord3> m_dirtyCell;
	
public:
	HWorldGrid(const std::string & name, Entity * parent = NULL);
	virtual ~HWorldGrid();
	
	bool insert(const std::string & name, const BoundingBox & box);
	bool insert(const std::string & name, const Coord3 & x);
/// insert with offset
	bool insert(const ValueType * v, const Vector3F & rel);
	bool insert(const ValueType * v, const Coord3 & x);
	bool insert(const float * at, const ValueType & v);
	void finishInsert();
	int elementSize();
/// override HBase
	virtual char save();
	virtual char load();
/// close all children
	virtual void close();
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
bool HWorldGrid<ChildType, ValueType>::insert(const std::string & name, const BoundingBox & box)
{
	std::cout<<"\n wg insert "<<name<<" "<<box;
	m_dirtyName = name;
	m_dirtyCell.clear();
/// find intesected grid
	const Coord3 lo = WorldGrid<ChildType, ValueType>::gridCoord((const float *)&box.getMin());
	const Coord3 hi = WorldGrid<ChildType, ValueType>::gridCoord((const float *)&box.getMax());
	int i, j, k;
	for(k=lo.z; k<= hi.z; ++k) {
		for(j=lo.y; j<= hi.y; ++j) {
			for(i=lo.x; i<= hi.x; ++i) {
				insert(name, Coord3(i,j,k) );
				m_dirtyCell.insert(Coord3(i,j,k) );
			}
		}
	}
	std::cout<<"\n affect "<<m_dirtyCell.size()<<" cells ";
	return true;
}

template<typename ChildType, typename ValueType>
bool HWorldGrid<ChildType, ValueType>::insert(const std::string & name, const Coord3 & x)
{
	Pair<Coord3, Entity> * p = Sequence<Coord3>::insert(x);
	if(!p->index) {
		p->index = new ChildType(childPath(coord3Str(x) ), this);
	}
	static_cast<ChildType *>(p->index)->insert(name);
	return true;
}

template<typename ChildType, typename ValueType>
bool HWorldGrid<ChildType, ValueType>::insert(const ValueType * v, 
												const Vector3F & rel) 
{
	BoundingBox box = v->calculateBBox();
	box.translate(rel);
/// find intesected grid
	const Coord3 lo = WorldGrid<ChildType, ValueType>::gridCoord((const float *)&box.getMin());
	const Coord3 hi = WorldGrid<ChildType, ValueType>::gridCoord((const float *)&box.getMax());
	int i, j, k;
	for(k=lo.z; k<= hi.z; ++k) {
		for(j=lo.y; j<= hi.y; ++j) {
			for(i=lo.x; i<= hi.x; ++i) {
				insert(v, Coord3(i,j,k));
			}
		}
	}
	return true;
}

template<typename ChildType, typename ValueType>
bool HWorldGrid<ChildType, ValueType>::insert(const ValueType * v, const Coord3 & x)
{
	Pair<Coord3, Entity> * p = Sequence<Coord3>::insert(x);
	if(!p->index) {
		//p->index = new ChildType(childPath(coord3Str(x) ), this);
		//static_cast<ChildType *>(p->index)->beginInsert();
		std::cout<<"\n error world grid has no cell "<<x;
		return false;
	}
	static_cast<ChildType *>(p->index)->insert(v);
	return true;
}

template<typename ChildType, typename ValueType>
bool HWorldGrid<ChildType, ValueType>::insert(const float * at, const ValueType & v) 
{
	const Coord3 x = WorldGrid<ChildType, ValueType>::gridCoord(at);
	
	Pair<Coord3, Entity> * p = Sequence<Coord3>::insert(x);
	if(!p->index) {
		p->index = new ChildType(childPath(coord3Str(x) ), this);
		static_cast<ChildType *>(p->index)->beginInsert();
	}
	static_cast<ChildType *>(p->index)->insert(at, (char *)&v);
	return true;
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
		if(cell)
			sum += cell->numElements();
		else 
			std::cout<<"\n warning world grid has no cell "<<WorldGrid<ChildType, ValueType>::key();
		
		WorldGrid<ChildType, ValueType>::next();
	}
	return sum;
}

template<typename ChildType, typename ValueType>
std::string HWorldGrid<ChildType, ValueType>::coord3Str(const Coord3 & c) const
{ return boost::str(boost::format("%1%,%2%,%3%") % c.x % c.y % c.z ); }

template<typename ChildType, typename ValueType>
char HWorldGrid<ChildType, ValueType>::save()
{
/// any dirty cells
	m_dirtyCell.begin();
	while(!m_dirtyCell.end() ) {
		Pair<Entity *, Entity> p = Sequence<Coord3>::findEntity(m_dirtyCell.key() );
		if(p.index) {
			static_cast<ChildType *>(p.index)->flush();
		}
		else 
			std::cout<<"\n error world grid has no cell "<<m_dirtyCell.key();
		
		m_dirtyCell.next();
	}
	
	HOocArray<hdata::TInt, 3, 256> cellCoords(".cells");

	if(hasNamedData(".cells") )
		cellCoords.openStorage(fObjectId, true);
	else
		cellCoords.createStorage(fObjectId);

	WorldGrid<ChildType, ValueType>::begin();
	while(!WorldGrid<ChildType, ValueType>::end() ) {
		Coord3 c = WorldGrid<ChildType, ValueType>::key();
		cellCoords.insert((char *)&c );
		WorldGrid<ChildType, ValueType>::next();
	}
	
	cellCoords.finishInsert();
	int n=cellCoords.numCols();
	
	WorldGrid<ChildType, ValueType>::calculateBBox();
	
	if(!hasNamedAttr(".bbx") )
	    addFloatAttr(".bbx", 6);
	writeFloatAttr(".bbx", (float *)&WorldGrid<ChildType, ValueType>::boundingBox() );
	
	float gz = WorldGrid<ChildType, ValueType>::gridSize();
	if(!hasNamedAttr(".gsz") )
	    addFloatAttr(".gsz", 1);
	writeFloatAttr(".gsz", &gz );
	
	if(!hasNamedAttr(".ncel") )
	    addIntAttr(".ncel", 1);
	writeIntAttr(".ncel", &n);
	
	int nelm = elementSize();
	if(!hasNamedAttr(".nelm") )
	    addIntAttr(".nelm", 1);
	writeIntAttr(".nelm", &nelm);
	
	if(!hasNamedAttr(".vlt") )
	    addIntAttr(".vlt", 1);
	writeIntAttr(".vlt", (int *)&ValueType::ShapeTypeId );
	
	std::cout<<"\n HWorldGrid save n value "<<nelm
	    <<"\n n cell "<<n
	    <<"\n grid size "<<gz
	    <<"\n bounding box "<<WorldGrid<ChildType, ValueType>::boundingBox()
	    <<"\n value type "<<ValueType::GetTypeStr();
	
	return 1;
}

template<typename ChildType, typename ValueType>
char HWorldGrid<ChildType, ValueType>::load()
{
	readFloatAttr(".bbx", (float *)WorldGrid<ChildType, ValueType>::boundingBoxR() );
	readFloatAttr(".gsz", WorldGrid<ChildType, ValueType>::gridSizeR() );
	
	HOocArray<hdata::TInt, 3, 256> cellCoords(".cells");
	cellCoords.openStorage(fObjectId);
	const int & ncoord = cellCoords.numCols();
	Coord3 c;
	for(int i=0; i<ncoord; ++i) {
	    cellCoords.readColumn((char *)&c, i);
	    Pair<Coord3, Entity> * p = Sequence<Coord3>::insert(c);
        if(!p->index) {
            p->index = new ChildType(childPath(coord3Str(c) ), this);
			static_cast<ChildType *>(p->index)->beginRead();
        }
	}
		
	std::cout<<"\n n cell "<<WorldGrid<ChildType, ValueType>::size()
	    <<"\n grid size "<<WorldGrid<ChildType, ValueType>::gridSize()
       <<"\n grid bbox "<<WorldGrid<ChildType, ValueType>::boundingBox()
       <<"\n n element "<<elementSize();
    return 1;
}

template<typename ChildType, typename ValueType>
void HWorldGrid<ChildType, ValueType>::close()
{
	WorldGrid<ChildType, ValueType>::begin();
	while(!WorldGrid<ChildType, ValueType>::end() ) {
		WorldGrid<ChildType, ValueType>::value()->close();
		WorldGrid<ChildType, ValueType>::next();
	}
	HBase::close();
}

}
}
#endif        //  #ifndef HWORLDGRID_H

