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
#include <MlFeather.h>
#include <CollisionRegion.h>
MlCluster::MlCluster() 
{
    m_sampleIndices = 0;
    m_angleStart = 0;
    m_angles = 0;
	m_sampleDirs = 0;
	m_sampleNSegs = 0;
	m_sampleLengths = 0;
	m_sampleBend = 0;
}

MlCluster::~MlCluster() 
{
    if(m_sampleIndices) delete[] m_sampleIndices;
    if(m_angleStart) delete[] m_angleStart;
    if(m_angles) delete[] m_angles;
	if(m_sampleDirs) delete[] m_sampleDirs;
	if(m_sampleNSegs) delete[] m_sampleNSegs;
	if(m_sampleLengths) delete[] m_sampleLengths;
	if(m_sampleBend) delete[] m_sampleBend;
}

void MlCluster::setK(unsigned k)
{
    KMeansClustering::setK(k);
    if(m_sampleIndices) delete[] m_sampleIndices;
    m_sampleIndices = new unsigned[k];
    if(m_angleStart) delete[] m_angleStart;
    m_angleStart = new unsigned[k];
	if(m_sampleDirs) delete[] m_sampleDirs;
	m_sampleDirs = new Vector3F[k];
	if(m_sampleNSegs) delete[] m_sampleNSegs;
	m_sampleNSegs = new short[k];
	if(m_sampleLengths) delete[] m_sampleLengths;
	m_sampleLengths = new float[k];
	if(m_sampleBend) delete[] m_sampleBend;
	m_sampleBend = new float[k];
}

void MlCluster::compute(MlCalamusArray * calamus, AccPatchMesh * mesh, unsigned begin, unsigned end)
{
	if(begin >= end) {
		setValid(0);
		return;
	}
	const unsigned n = end - begin;
	unsigned i;
	setN(n);
	if(n < 3) {
		setK(n);
		resetGroup();
		for(i = 0; i < N(); i++) m_sampleIndices[i] = begin + i;
		createAngles(calamus);
		setValid(1);
		return;
	}
	const unsigned k = 2 + (n - 1) / 3;
	setK(k);
	unsigned j;
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
	
	for(unsigned i = 0; i < K(); i++)
		assignGroupSample(calamus, mesh, begin, i);
	createAngles(calamus);
	computeSampleDirs(calamus, mesh);
	setValid(1);
}

void MlCluster::assignGroupSample(MlCalamusArray * calamus, AccPatchMesh * mesh, unsigned begin, unsigned grp)
{
	Vector3F pos;
	float d, minD = 10e8;
	for(unsigned i = 0; i < N(); i++) {
		if(group(i) != grp) continue;
		MlCalamus * c = calamus->asCalamus(begin + i);
		mesh->pointOnPatch(c->faceIdx(), c->patchU(), c->patchV(), pos);
		d = Vector3F(pos, groupCenter(grp)).length();
		if(d < minD) {
			minD = d;
			m_sampleIndices[grp] = begin + i;
		}
	}
}

void MlCluster::createAngles(MlCalamusArray * calamus)
{
	unsigned numAngles = 0;
	for(unsigned i = 0; i < K(); i++) {
		m_angleStart[i] = numAngles;
		MlCalamus * c = calamus->asCalamus(m_sampleIndices[i]);
		m_sampleNSegs[i] = c->feather()->numSegment();
		numAngles += m_sampleNSegs[i];
	}
	
	if(m_angles) delete[] m_angles;
	m_angles = new float[numAngles];
}

void MlCluster::computeSampleDirs(MlCalamusArray * calamus, AccPatchMesh * mesh)
{
	Matrix33F tang, space;
	for(unsigned i = 0; i < K(); i++) {
		MlCalamus * c = calamus->asCalamus(m_sampleIndices[i]);
		mesh->tangentFrame(c->faceIdx(), c->patchU(), c->patchV(), tang);
		space.setIdentity();
		space.rotateX(c->rotateX());
		space.multiply(tang);
		m_sampleDirs[i] = space.transform(Vector3F::ZAxis);
		m_sampleLengths[i] = c->realScale();
	}
}

float * MlCluster::angles(unsigned idx) const
{
    return &m_angles[m_angleStart[idx]];
}

unsigned MlCluster::sampleIdx(unsigned idx) const
{
	return m_sampleIndices[idx];
}

void MlCluster::recordAngles(MlCalamus * c, unsigned idx)
{
	float * dst = angles(idx);
	float * src = c->feather()->angles();
	const short ns = c->featherNumSegment();
	for(short i = 0; i < ns; i++) dst[i] = src[i];
	
	m_sampleBend[idx] = c->feather()->bendDirection();
}

void MlCluster::reuseAngles(MlCalamus * c, unsigned idx)
{
	float * src = angles(idx);
	float * dst = c->feather()->angles();
	const short ns = c->featherNumSegment();
	for(short i = 0; i < ns; i++) dst[i] = src[i];
}

short MlCluster::sampleNSeg(unsigned idx) const
{
	return m_sampleNSegs[idx];
}

Vector3F MlCluster::sampleDir(unsigned idx) const
{
	return m_sampleDirs[idx];
}

float MlCluster::sampleLength(unsigned idx) const
{
	return m_sampleLengths[idx];
}

float MlCluster::sampleBend(unsigned idx) const
{
	return m_sampleBend[idx];
}
