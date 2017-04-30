/*
 *  HWorldGrid.cpp
 *  
 *
 *  Created by jian zhang on 3/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "HWorldGrid.h"

namespace aphid {

namespace sdb {

HVarGrid::HVarGrid(const std::string & name) : HBase(name)
{ m_vtyp = 0; }

HVarGrid::~HVarGrid()
{}

char HVarGrid::verifyType()
{
    if(!hasNamedAttr(".bbx"))
		return 0;
	if(!hasNamedAttr(".gsz"))
		return 0;
	if(!hasNamedData(".cells"))
		return 0;
	if(!hasNamedAttr(".ncel"))
		return 0;
	if(!hasNamedAttr(".nelm"))
		return 0;
	if(!hasNamedAttr(".vlt"))
		return 0;
    return 1;
}

char HVarGrid::load()
{
    readIntAttr(".vlt", &m_vtyp);
    readFloatAttr(".gsz", &m_gsize);
    readFloatAttr(".bbx", m_bbx);
    return 1;
}

const int & HVarGrid::valueType() const
{ return m_vtyp; }

const float & HVarGrid::gridSize() const
{ return m_gsize; }

const float * HVarGrid::bbox() const
{ return m_bbx; }

}

}