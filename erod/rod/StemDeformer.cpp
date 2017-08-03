/*
 *  StemDeformer.cpp
 *
 *  Created by jian zhang on 8/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "StemDeformer.h"

using namespace aphid;

StemDeformer::StemDeformer() :
m_numPoints(0)
{}

StemDeformer::~StemDeformer()
{}

void StemDeformer::createDeformer(const int& np, const int& ns)
{
	m_localPnt.reset(new Vector3F[np]);
	m_dfmdPnt.reset(new Vector3F[np]);
	m_bindInd.reset(new int[np]);
	m_numPoints = np;
	m_tm.reset(new Matrix44F[ns]);
	m_numSpaces = ns;
}

bool StemDeformer::solve()
{ 
	for(int i=0;i<m_numPoints;++i) {
		m_dfmdPnt[i] = m_tm[m_bindInd[i]].transform(m_localPnt[i]);
	}
	return true; 
}

const int& StemDeformer::numPoints() const
{ return m_numPoints; }

const int& StemDeformer::numSpaces() const
{ return m_numSpaces; }

const aphid::Vector3F* StemDeformer::deformedPnt() const
{ return m_dfmdPnt.get(); }

int* StemDeformer::bindInds()
{ return m_bindInd.get(); }

Matrix44F* StemDeformer::spaces()
{ return m_tm.get(); }
