#ifndef BVHSOLVER_H
#define BVHSOLVER_H

/*
 *  BvhSolver.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <BaseSolverThread.h>
#include <bvh_common.h>
#include <radixsort_implement.h>

class BaseBuffer;
class CUDABuffer;
class BvhTriangleMesh;

class BvhSolver : public BaseSolverThread
{
public:
	BvhSolver(QObject *parent = 0);
	virtual ~BvhSolver();
	
	void init();
	void setMesh(BvhTriangleMesh * mesh);
	void createEdges(BaseBuffer * onhost, uint n);
	void createRays(uint m, uint n);
	
	const unsigned numPoints() const;
	const unsigned numLeafNodes() const;
	const unsigned numInternalNodes() const;
	const unsigned numRays() const;
	
	void setAlpha(float x);
	void setPlaneUDim(uint x);
	
	int getRootNodeIndex();
	void getRootNodeAabb(Aabb * dst); 
	void getPoints(BaseBuffer * dst);
	void getRays(BaseBuffer * dst);	
	void getLeafAabbs(BaseBuffer * dst);
	void getInternalAabbs(BaseBuffer * dst);
	void getLeafHash(BaseBuffer * dst);
	void getInternalDistances(BaseBuffer * dst);
	void getInternalChildIndex(BaseBuffer * dst);

protected:
    virtual void stepPhysics(float dt);	
private:
	void formLeafAabbs();
	void combineAabb();
	void calcLeafHash();
	void buildInternalTree();
	void findMaxDistanceFromRoot();
	void formInternalTreeAabbsIterative();
	void formRays();
	void rayTraverse();
	
	void printLeafInternalNodeConnection();
	void printInternalNodeConnection();
	
private:
	BvhTriangleMesh * m_mesh;
	CUDABuffer * m_edgeContactIndices;
	CUDABuffer * m_leafAabbs;
	CUDABuffer * m_internalNodeAabbs;
	CUDABuffer * m_leafHash[2];
	CUDABuffer * m_internalNodeCommonPrefixValues;
	CUDABuffer * m_internalNodeCommonPrefixLengths;
	CUDABuffer * m_leafNodeParentIndices;
	CUDABuffer * m_internalNodeChildIndices;
	CUDABuffer * m_internalNodeParentIndices;
	CUDABuffer * m_rootNodeIndexOnDevice;
    CUDABuffer * m_distanceInternalNodeFromRoot;
	CUDABuffer * m_reducedMaxDistance;
	CUDABuffer * m_rays;
	CUDABuffer * m_ntests;
    
	unsigned m_numLeafNodes, m_numRays, m_rayDim;
	float m_alpha;
};
#endif        //  #ifndef BVHSOLVER_H
