/*
 *  GeodesicSkeleton.cpp
 *  
 *
 *  Created by jian zhang on 10/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GeodesicSkeleton.h"

namespace aphid {

namespace topo {

GeodesicSkeleton::GeodesicSkeleton() :
m_numJoints(0)
{}

GeodesicSkeleton::~GeodesicSkeleton()
{}

const int& GeodesicSkeleton::numJoints() const
{ return m_numJoints; }

const Vector3F* GeodesicSkeleton::jointPos() const
{ return m_jointPos.get(); }

void GeodesicSkeleton::clearAllJoint()
{ m_numJoints = 0; }

bool GeodesicSkeleton::buildSkeleton(const float& unitD,
					const Vector3F* pos)
{
	const int np = numRegions();
	PathData* pds = new PathData[np];
	
	bool stat = buildLevelSet(pds, unitD, pos);
	
	if(stat)
		stat = buildJoints(pds);
	else
		m_numJoints = 0;
	
	delete[] pds;
	
	if(stat)
		connectPieces();
	
	return stat;
}

bool GeodesicSkeleton::buildJoints(PathData* pds)
{
	const int np = numRegions();
	m_pieceCounts.reset(new int[np]);
	m_pieceBegins.reset(new int[np + 1]);
	m_pieceParent.reset(new int[np]);
	
	m_numJoints = 0;
	for(int i=0;i<np;++i) {
		const int ns = pds[i].numSets();
		m_pieceCounts[i] = ns;
		m_pieceBegins[i] = m_numJoints;
		m_pieceParent[i] = -1;
		m_numJoints += ns;
	}
	m_pieceBegins[np] = m_numJoints;
	
	m_jointPos.reset(new Vector3F[m_numJoints]);
	for(int i=0;i<np;++i) {
		pds[i].getSetPos(&m_jointPos[m_pieceBegins[i] ]);
	}
	
	return true;
}

void GeodesicSkeleton::connectPieces()
{
	const float* dist = (const float*)distanceToRoot();
	const int* pathI = vertexPathInd();
	const int* jointI = vertexSetInd();
	
	const int& nv = numVertices();
	for(int i=0;i<nv;++i) {
		if(jointI[i] > 0)
			continue;
/// first set in piece
		int neiJ = getLowestNeightInd(dist, i);
		
		const int& childPieceI = pathI[i];
/// path changes and dist is lower			
		if(dist[neiJ] < dist[i] 
			&& pathI[neiJ] != childPieceI) {
			
			bool stat = m_pieceParent[childPieceI] < 0;
			if(!stat)
				stat = getPieceVaryingJointIndex(m_pieceParent[childPieceI]) > jointI[neiJ];
			
			if(stat) {
				
				std::cout<<"\n connect piece "<<childPieceI
					<<" to piece"<<pathI[neiJ]<<" joint"<<jointI[neiJ];
					
				m_pieceParent[childPieceI] = getJointIndex(pathI[neiJ], jointI[neiJ]);
			}		
		}
	}
}

int GeodesicSkeleton::getJointIndex(const int& pieceI, const int& jointJ) const
{ return m_pieceBegins[pieceI] + jointJ; }

int GeodesicSkeleton::getPieceVaryingJointIndex(const int& x) const
{
	const int np = numRegions();
	for(int i=0;i<np;++i) {
		if(x >= m_pieceBegins[i] 
			&& x < m_pieceBegins[i+1])
			return x - m_pieceBegins[i];
	}
	return 0;
}

}

}