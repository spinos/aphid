/*
 *  SkeletonSubspaceDeformer.h
 *  aphid
 *
 *  Created by jian zhang on 10/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseDeformer.h>

class SkeletonSystem;

class SkeletonSubspaceDeformer : public BaseDeformer {
public:
	SkeletonSubspaceDeformer();
	virtual ~SkeletonSubspaceDeformer();
	virtual void clear();
	virtual void setMesh(BaseMesh * mesh);
	virtual char solve();
	
	void bindToSkeleton(SkeletonSystem * skeleton);

protected:
	void bindVertexToSkeleton(unsigned vi, std::vector<float> & wei);
	void calculateSubspaceP();
private:
	VectorN<unsigned> * m_jointIds;
	float * m_jointWeights;
	Vector3F * m_subspaceP;
	SkeletonSystem * m_skeleton;
};