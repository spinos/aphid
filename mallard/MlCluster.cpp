/*
 *  MlCluster.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlCluster.h"
#include "MlCalamusArray.h"
#include <AccPatchMesh.h>
#include <mlFeather.h>
MlCluster::MlCluster() 
{
    m_featherIndices = 0;
    m_angleStart = 0;
    m_angles = 0;
}

MlCluster::~MlCluster() 
{
    if(m_featherIndices) delete[] m_featherIndices;
    if(m_angleStart) delete[] m_angleStart;
    if(m_angles) delete[] m_angles;
}

void MlCluster::setK(unsigned k)
{
    KMeansClustering::setK(k);
    if(m_featherIndices) delete[] m_featherIndices;
    m_featherIndices = new short[k];
    for(unsigned i = 0; i < k; i++) m_featherIndices[i] = -1;
    if(m_angleStart) delete[] m_angleStart;
    m_angleStart = new unsigned[k];
}

void MlCluster::setN(unsigned n)
{
    KMeansClustering::setN(n);
}

void MlCluster::compute(MlCalamusArray * calamus, AccPatchMesh * mesh, unsigned begin, unsigned end)
{
	if(begin >= end) return;
	const unsigned n = end - begin;
	setN(n);
	if(n < 5) {
		setK(n);
		resetGroup();
		return;
	}
	const unsigned k = 4 + (n - 1) / 5;
	setK(k);
	unsigned i, j;
	float d;
	Vector3F pos;
	const unsigned faceIdx = calamus->asCalamus(begin)->faceIdx();
	for(i = 0; i < k; i++) {
		MlCalamus * c = calamus->asCalamus(begin + i);
		mesh->pointOnPatch(faceIdx, c->patchU(), c->patchV(), pos);
		setInitialGuess(i, pos);
	}
	
	for(j = 0; j < 8; j++) {
		preAssign();
		for(i = begin; i < end; i++) {
			MlCalamus * c = calamus->asCalamus(i);
			mesh->pointOnPatch(faceIdx, c->patchU(), c->patchV(), pos);
			assignToGroup(i - begin, pos);
		}
		d = moveCentroids();
		if(d < 10e-3) break;
	}
	
	unsigned numAngles = 0;
	for(i = 0; i < k; i++) {
		MlCalamus * c = calamus->asCalamus(begin + i);
		if(m_featherIndices[i] < 0) {
		    m_featherIndices[i] = c->featherIdx();
		    m_angleStart[i] = numAngles;
		    numAngles += c->feather()->numSegment();
		}
	}
	
	if(m_angles) delete[] m_angles;
	m_angles = new float[numAngles];
}

float * MlCluster::angles(unsigned idx) const
{
    return &m_angles[m_angleStart[idx]];
}

void MlCluster::computeAngles()
{
    
}