/*
 *  SkeletonSubspaceDeformer.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "SkeletonSubspaceDeformer.h"
#include <SkeletonSystem.h>
#include <SkeletonJoint.h>
SkeletonSubspaceDeformer::SkeletonSubspaceDeformer() 
{
	m_skeleton = 0;
	m_jointIds = 0;
	m_subspaceP = 0;
}

SkeletonSubspaceDeformer::~SkeletonSubspaceDeformer() {}

void SkeletonSubspaceDeformer::clear()
{
	if(m_jointIds) delete[] m_jointIds;
	if(m_subspaceP) delete[] m_subspaceP;
	m_jointIds = 0;
	m_subspaceP = 0;
	BaseDeformer::clear();
}

void SkeletonSubspaceDeformer::setMesh(BaseMesh * mesh)
{
	BaseDeformer::setMesh(mesh);
	m_jointIds = new unsigned[numVertices()];
	m_subspaceP = new Vector3F[numVertices()];
}

void SkeletonSubspaceDeformer::bindToSkeleton(SkeletonSystem * skeleton)
{
	m_skeleton = skeleton;
	unsigned i, j;
	Matrix44F spaceInv;
	Vector3F * p = getDeformedP();
	for(i = 0; i < numVertices(); i++) {
		j = skeleton->closestJointIndex(p[i]);
		m_jointIds[i] = j;
		SkeletonJoint * joint = skeleton->jointByIndex(j);
		spaceInv = joint->worldSpace();
		spaceInv.inverse();
		m_subspaceP[i] = spaceInv.transform(p[i]);
	}
}

char SkeletonSubspaceDeformer::solve()
{
	if(!m_skeleton) return 0;
	unsigned i, j;
	Matrix44F space;
	Vector3F * p = deformedP();
	for(i = 0; i < numVertices(); i++) {
		j = m_jointIds[i];
		SkeletonJoint * joint = m_skeleton->jointByIndex(j);
		space = joint->worldSpace();
		p[i] = space.transform(m_subspaceP[i]);
	}
	return 1;
}
