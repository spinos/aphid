/*
 *  BvhSolver.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BaseSolverThread.h>
#include <bvh_common.h>
#include <radixsort_implement.h>
#include <app_define.h>

class BaseBuffer;
class CUDABuffer;

class BvhSolver : public BaseSolverThread
{
public:
	BvhSolver(QObject *parent = 0);
	virtual ~BvhSolver();
	
	void init();
	
	const unsigned numVertices() const;
	
	unsigned getNumTriangleFaceVertices() const;
	unsigned * getIndices() const;
	const unsigned numTriangles() const;
	const unsigned numEdges() const;
	float * displayVertex();
	EdgeContact * edgeContacts();
	void setAlpha(float x);
	const Aabb combinedAabb() const; 
	
	const unsigned numLeafNodes() const;
	const unsigned numInternalNodes() const;
	
#ifdef BVHSOLVER_DBG_DRAW
	Aabb * displayLeafAabbs();
	Aabb * displayInternalAabbs();
	KeyValuePair * displayLeafHash();
#endif

protected:
    virtual void stepPhysics(float dt);	
private:
	void formPlane(float alpha);
	void formLeafAabbs();
	void combineAabb();
	void calcLeafHash();
	void buildInternalTree();
	void findMaxDistanceFromRoot();
	void formInternalTreeAabbsIterative();
	
	void printLeafInternalNodeConnection();
	void printInternalNodeConnection();
	
private:
	Aabb m_bigAabb;
	BaseBuffer * m_displayVertex;
	CUDABuffer * m_vertexBuffer;
	CUDABuffer * m_edgeContactIndices;
	unsigned * m_triIndices;
	BaseBuffer * m_edges;
	CUDABuffer * m_leafAabbs;
	CUDABuffer * m_internalNodeAabbs;
	BaseBuffer * m_lastReduceBlk;
	CUDABuffer * m_leafHash[2];
	CUDABuffer * m_internalNodeCommonPrefixValues;
	CUDABuffer * m_internalNodeCommonPrefixLengths;
	CUDABuffer * m_leafNodeParentIndices;
	CUDABuffer * m_internalNodeChildIndices;
	CUDABuffer * m_internalNodeParentIndices;
	CUDABuffer * m_rootNodeIndexOnDevice;
    CUDABuffer * m_distanceInternalNodeFromRoot;
	CUDABuffer * m_reducedMaxDistance;
    
#ifdef BVHSOLVER_DBG_DRAW
	BaseBuffer * m_displayLeafAabbs;
	BaseBuffer * m_displayInternalAabbs;
	BaseBuffer * m_displayLeafHash;
#endif
	int m_rootNodeIndex;
	unsigned m_numTriIndices, m_numTriangles, m_numEdges;
	float m_alpha;
};