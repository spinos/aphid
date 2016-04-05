/*
 *  HInnerGrid.h
 *  julia
 *
 *  Created by jian zhang on 1/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  child of world grid
 */

#pragma once

#include <HBase.h>
#include <HOocArray.h>
#include <BoundingBox.h>
#include "Entity.h"

namespace aphid {
namespace sdb {

template <int DataRank, int NRows, int BufSize>
class HInnerGrid : public HBase, public Entity {
	
	HOocArray<DataRank, NRows, BufSize> *m_data;
	
public:
	HInnerGrid(const std::string & name, Entity * parent);
	virtual ~HInnerGrid();
	
	bool insert(const float * at, char * v);
	void finishInsert();
	int numElements();
	void beginInsert();
	bool beginRead();
	void flush();
	void buildTree(const BoundingBox & rootBox);
	bool isEmpty();
	void getBBox(BoundingBox * dst);
	void getNumVoxel(int * dst);

protected:

private:

};

template <int DataRank, int NRows, int BufSize>
HInnerGrid<DataRank, NRows, BufSize>::HInnerGrid(const std::string & name, Entity * parent) :
HBase(name), Entity(parent),
m_data(NULL)
{}

template <int DataRank, int NRows, int BufSize>
HInnerGrid<DataRank, NRows, BufSize>::~HInnerGrid()
{
	if(m_data) delete m_data;
}

template <int DataRank, int NRows, int BufSize>
int HInnerGrid<DataRank, NRows, BufSize>::numElements()
{
	if(!m_data) return 0;
	return m_data->numCols();
}

template <int DataRank, int NRows, int BufSize>
bool HInnerGrid<DataRank, NRows, BufSize>::insert(const float * at, char * v) 
{
	m_data->insert(v);
	return true;
}

template <int DataRank, int NRows, int BufSize>
void HInnerGrid<DataRank, NRows, BufSize>::finishInsert()
{ m_data->finishInsert(); }

template <int DataRank, int NRows, int BufSize>
void HInnerGrid<DataRank, NRows, BufSize>::beginInsert()
{
	if(!m_data)
		m_data = new HOocArray<DataRank, NRows, BufSize>(".data");
		
	if(hasNamedData(".data") )
		m_data->openStorage(fObjectId, true);
	else 
		m_data->createStorage(fObjectId);
}

template <int DataRank, int NRows, int BufSize>
bool HInnerGrid<DataRank, NRows, BufSize>::beginRead()
{
	if(!hasNamedData(".data") ) return false;
	if(!m_data)
		m_data = new HOocArray<DataRank, NRows, BufSize>(".data");
	m_data->openStorage(fObjectId);
	return true;
}

template <int DataRank, int NRows, int BufSize>
void HInnerGrid<DataRank, NRows, BufSize>::flush()
{}

template <int DataRank, int NRows, int BufSize>
void HInnerGrid<DataRank, NRows, BufSize>::buildTree(const BoundingBox & rootBox)
{}

template <int DataRank, int NRows, int BufSize>
bool HInnerGrid<DataRank, NRows, BufSize>::isEmpty()
{ return numElements() < 1; }

template <int DataRank, int NRows, int BufSize>
void HInnerGrid<DataRank, NRows, BufSize>::getBBox(BoundingBox * dst)
{}

template <int DataRank, int NRows, int BufSize>
void HInnerGrid<DataRank, NRows, BufSize>::getNumVoxel(int * dst)
{}

}
}
