/*
 *  BlueYellowCyanRefine.cpp
 *  
 *
 *  Created by jian zhang on 7/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BlueYellowCyanRefine.h"

namespace ttg {

BlueYellowCyanRefine::BlueYellowCyanRefine(int vred, int vyellow, int vblue, int vcyan) 
{
	setTetrahedronVertices(m_tet[0], vred, vyellow, vblue, vcyan);
	m_N = 1;
}

const int & BlueYellowCyanRefine::numTetra() const
{ return m_N; }

const ITetrahedron * BlueYellowCyanRefine::tetra(int i) const
{ return &m_tet[i]; }

void BlueYellowCyanRefine::splitYellow(int vyellow)
{
	ITetrahedron * oldTet = lastTetra();
	setTetrahedronVertices(m_tet[m_N], oldTet->iv0, oldTet->iv1, oldTet->iv2, oldTet->iv3);
	
	oldTet->iv0 = vyellow;
	m_tet[m_N].iv1 = vyellow;
	m_N++;
}

void BlueYellowCyanRefine::splitBlue(int vblue)
{
	ITetrahedron * oldTet = lastTetra();
	setTetrahedronVertices(m_tet[m_N], oldTet->iv0, oldTet->iv1, oldTet->iv2, oldTet->iv3);
	
	oldTet->iv0 = vblue;
	m_tet[m_N].iv2 = vblue;
	m_N++;
}

void BlueYellowCyanRefine::splitCyan(int vcyan)
{
	ITetrahedron * oldTet = lastTetra();
	setTetrahedronVertices(m_tet[m_N], oldTet->iv0, oldTet->iv1, oldTet->iv2, oldTet->iv3);
	
	oldTet->iv0 = vcyan;
	m_tet[m_N].iv3 = vcyan;
	m_N++;
}

ITetrahedron * BlueYellowCyanRefine::lastTetra()
{ return &m_tet[m_N-1]; }

}