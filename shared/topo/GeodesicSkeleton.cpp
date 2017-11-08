/*
 *  GeodesicSkeleton.cpp
 *  
 *
 *  Created by jian zhang on 10/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GeodesicSkeleton.h"
#include "JointPiece.h"
#include <math/kmean.h>

namespace aphid {

namespace topo {

GeodesicSkeleton::GeodesicSkeleton() :
m_numPieces(0)
{}

GeodesicSkeleton::~GeodesicSkeleton()
{}

const int& GeodesicSkeleton::numPieces() const
{ return m_numPieces; }

void GeodesicSkeleton::clearAllJoint()
{ m_numPieces = 0; }

bool GeodesicSkeleton::buildSkeleton()
{
	m_numPieces = numRegions();
	m_pieces.reset(new JointPiece[m_numPieces] );
	for(int i=0;i<m_numPieces;++i) {
		std::vector<int > vertexSet;
		collectRegionVertices(vertexSet, i);
		buildClusters(vertexSet, i);
		vertexSet.clear();
	}
	connectPieces();
	std::cout.flush();
	return true;
}

void GeodesicSkeleton::buildClusters(const std::vector<int >& vertexSet,
						const int& jregion)
{
	const int n = vertexSet.size();
	int k = n>>6;
	if(k < 4)
		k = 4;
	if(k > 15)
		k = 15;
	std::cout<<"\n region"<<jregion<<" has nv "<<n;
		
/// kmean by euclidean distance 
	KMeansClustering2<float> cluster;
	cluster.setKND(k, n, 3);
	
/// build data
	DenseMatrix<float > d(n, 3);
	for(int i=0;i<n;++i) {
		const int& vi = vertexSet[i];
		const Vector3F& pv = vertexPos()[vi];
		d.column(0)[i] = pv.x;
		d.column(1)[i] = pv.y;
		d.column(2)[i] = pv.z;
	}
	
	cluster.compute(d);
	
	DenseVector<float> cent;
	const int& nj = cluster.K();
	m_pieces[jregion].create(nj);
	
	for(int i=0;i<nj;++i) {
		cluster.getGroupCentroid(cent, i);
		JointData& aj = m_pieces[jregion].joints()[i];
		cent.extractData(aj._posv);
	}
	
	m_pieces[jregion].zeroJointVal();
	
	const float* dist = distanceToSite(jregion);
	
	const int* b = cluster.groupIndices();
	for(int i=0;i<n;++i) {
		const int& vi = vertexSet[i];
		vertexSetInd()[vi] = b[i];
		m_pieces[jregion].addJointVal(dist[vi], b[i]);
	}
	
	m_pieces[jregion].averageJointVal();

	m_pieces[jregion].connectJoints(jregion > 0);
	
	for(int i=0;i<n;++i) {
		const int& vi = vertexSet[i];
		m_pieces[jregion].setJointInd(vertexSetInd()[vi]);
	}
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
		const int& parentPieceI = pathI[neiJ];
/// path changes and dist is lower			
		if(dist[neiJ] < dist[i] 
			&& parentPieceI != childPieceI) {
			
			bool stat = m_pieces[childPieceI].hasParent();

			if(!stat) {
				
				std::cout<<"\n connect piece "<<childPieceI
					<<" to piece"<<parentPieceI<<" joint"<<jointI[neiJ];
				
				JointData* pj = &m_pieces[parentPieceI].joints()[jointI[neiJ] ];
				m_pieces[childPieceI].setParent(pj);
				
			}
		}
	}
}

const JointPiece& GeodesicSkeleton::getPiece(const int& i) const
{ return m_pieces[i]; }

}

}