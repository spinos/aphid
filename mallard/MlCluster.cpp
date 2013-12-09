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
    m_sampleIndices = 0;
    m_angleStart = 0;
    m_angles = 0;
}

MlCluster::~MlCluster() 
{
    if(m_sampleIndices) delete[] m_sampleIndices;
    if(m_angleStart) delete[] m_angleStart;
    if(m_angles) delete[] m_angles;
}

void MlCluster::setK(unsigned k)
{
    KMeansClustering::setK(k);
    if(m_sampleIndices) delete[] m_sampleIndices;
    m_sampleIndices = new int[k];
    for(unsigned i = 0; i < k; i++) m_sampleIndices[i] = -1;
    if(m_angleStart) delete[] m_angleStart;
    m_angleStart = new unsigned[k];
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
		if(m_sampleIndices[i] < 0) {
		    m_sampleIndices[i] = begin + i;
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

void MlCluster::computeAngles(MlCalamusArray * calamus)
{
    unsigned i, nk = K();
	for(i = 0; i < nk; i++) {
		MlCalamus * c = calamus->asCalamus(m_sampleIndices[i]);
	}
}