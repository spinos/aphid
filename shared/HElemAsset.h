/*
 *  HElemAsset.h
 *  qef
 *
 *  Created by jian zhang on 3/23/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  large number of typed elements in a 2d array
 *  each element is a column
 *  with bounding box
 */
#pragma once
#include <HBase.h>
#include <HOocArray.h>
#include <Boundary.h>
#include <ConvexShape.h>
#include <VectorArray.h>

namespace aphid {

class HElemBase : public HBase {

public:
	HElemBase(const std::string & name);
	virtual ~HElemBase();
	
	virtual char verifyType();

};

template<typename T, int NRow>
class HElemAsset : public HElemBase, public Boundary {

	HOocArray<hdata::TChar, NRow, 1024> * m_data;
	int m_numElem;
	
public:
	HElemAsset(const std::string & name);
	virtual ~HElemAsset();
	
	virtual char save();
	virtual char load();
    
	void insert(const T & x);
	void extract(sdb::VectorArray<T> * dst);
	void clear();
	
	const int & numElems() const;
	
};

template<typename T, int NRow>
HElemAsset<T, NRow>::HElemAsset(const std::string & name) :
HElemBase(name),
m_data(NULL),
m_numElem(0)
{}

template<typename T, int NRow>
HElemAsset<T, NRow>::~HElemAsset()
{ if(m_data) delete m_data; }

template<typename T, int NRow>
char HElemAsset<T, NRow>::save()
{
	if(!m_data) {
		std::cout<<"\n helemasset has no data";
		return 0;
	}
	
	m_data->finishInsert();
	
	if(!hasNamedAttr(".bbx") )
	    addFloatAttr(".bbx", 6);
	writeFloatAttr(".bbx", (float *)&Boundary::getBBox() );
	
	if(!hasNamedAttr(".elemtyp") )
	    addIntAttr(".elemtyp", 1);
		
	int et = T::ShapeTypeId;
	writeIntAttr(".elemtyp", (int *)&et );
	
	if(!hasNamedAttr(".nelem") )
	    addIntAttr(".nelem", 1);
	
	m_numElem = m_data->numCols();
	writeIntAttr(".nelem", (int *)&m_numElem );
	
	std::cout<<"\n helemasset saved "<<m_numElem<<" "<<T::GetTypeStr();
	
	return 1;
}

template<typename T, int NRow>
void HElemAsset<T, NRow>::insert(const T & x)
{
	if(!m_data) {
		m_data = new HOocArray<hdata::TChar, NRow, 1024>(".data");
		
		if(hasNamedData(".data") )
			m_data->openStorage(fObjectId, true);
		else
			m_data->createStorage(fObjectId);
	}
	
	m_data->insert((char *)&x);
}

template<typename T, int NRow>
char HElemAsset<T, NRow>::load()
{
	if(!hasNamedAttr(".nelem") ) {
		std::cout<<"\n helemasset has not nelem attr";
		return 0;
	}
	
	BoundingBox b;
	readFloatAttr(".bbx", (float *)&b );
	setBBox(b);
	
	readIntAttr(".nelem", &m_numElem );
	
	m_data = new HOocArray<hdata::TChar, NRow, 1024>(".data");
	if(!m_data->openStorage(fObjectId)) {
		std::cout<<"\n helemasset cannot open data storage";
		return 0;
	}
	
	return 1;
}

template<typename T, int NRow>
const int & HElemAsset<T, NRow>::numElems() const
{ return m_numElem; }

template<typename T, int NRow>
void HElemAsset<T, NRow>::extract(sdb::VectorArray<T> * dst)
{
	if(!m_data) return;
	
	T s;
	int i=0;
	for(;i<m_numElem;++i) {
		m_data->readColumn((char *)&s, i);
		dst->insert(s);
	}
	
}

template<typename T, int NRow>
void HElemAsset<T, NRow>::clear()
{
	if(load() ) {
		m_data->reset();
		save();
	}
}

typedef HElemAsset<cvx::Triangle, 48> HTriangleAsset;

}