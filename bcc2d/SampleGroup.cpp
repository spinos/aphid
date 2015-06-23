/*
 *  SampleGroup.cpp
 *  testbcc
 *
 *  Created by jian zhang on 6/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "SampleGroup.h"

SampleGroup::SampleGroup() 
{
	m_groupSize = 0;
    m_groupCentroid = 0;
    m_numGroups = 0;
}

SampleGroup::~SampleGroup() 
{
	if(m_groupSize) delete[] m_groupSize;
    if(m_groupCentroid) delete[] m_groupCentroid;
}

float * SampleGroup::groupSize()
{ return m_groupSize; }

Vector3F * SampleGroup::groupCentroid()
{ return m_groupCentroid; }

void SampleGroup::createGroupSize(unsigned n)
{
    if(m_groupSize) delete[] m_groupSize;
	m_groupSize = new float[n];
    m_numGroups = n;
}

void SampleGroup::createGroupCentroid(unsigned n)
{
    if(m_groupCentroid) delete[] m_groupCentroid;
	m_groupCentroid = new Vector3F[n];
}

void SampleGroup::compute(Vector3F * samples, unsigned numSamples, unsigned numGroups)
{
    if(numSamples < numGroups) return;
    
    createGroupSize(numGroups);
    createGroupCentroid(numGroups);
    
    float * gs = groupSize();
    Vector3F * gc = groupCentroid();
    unsigned i, j;
    for(i=0; i < numGroups; i++) gs[i] = 0.f;
    for(i=0; i < numGroups; i++) gc[i].setZero();
    
    unsigned * counts = new unsigned[numGroups];
    for(i=0; i < numGroups; i++) counts[i] = 0;
    
    float fcpg = (float)numSamples / (float)m_numGroups;
	unsigned cpg = fcpg;
	if(fcpg - cpg > .6f) cpg++;
    
    unsigned igrp;
	for(i=0; i<numSamples; i++) {
		igrp = i / cpg;
		if(igrp > numGroups-1) igrp = numGroups -1;
		gc[igrp] += samples[i];
		counts[igrp]++;
	}
    
    for(i=0; i<numGroups; i++) {
		gc[i] *= 1.f/(float)counts[i];
		std::cout<<" count in group"<<i<<" "<<counts[i]<<"\n";
	}
    
    float d;
    for(i=0; i<numSamples; i++) {
		igrp = i / cpg;
		if(igrp > numGroups-1) igrp = numGroups -1;
		d = gc[igrp].distanceTo(samples[i]);
		if(gs[igrp] < d) gs[igrp] = d;
	}
    
    delete[] counts;
}

unsigned SampleGroup::numGroups() const
{ return m_numGroups; }

KMeanSampleGroup::KMeanSampleGroup() {}
KMeanSampleGroup::~KMeanSampleGroup() {}
void KMeanSampleGroup::compute(Vector3F * samples, unsigned numSamples, unsigned numGroups)
{
	setValid(0);
	
	if(numSamples < numGroups) return;
	
	setN(numSamples);
	setK(numGroups);
	
	const unsigned seg = numSamples / numGroups;
	float d;
	unsigned i, j;
	for(i = 0; i < numGroups; i++) {
		setInitialGuess(i, samples[i*seg]);
	}
	
	for(j = 0; j < numSamples*3; j++) {
		preAssign();
		for(i = 0; i < numSamples; i++) {
			assignToGroup(i, samples[i]);
		}
		d = moveCentroids();
		if(d < 10e-3) break;
	}
	
	// std::cout<<" j end "<<j;
	float * gs = groupSize();
    
	for(i = 0; i < K(); i++) gs[i] = 0.f;
	
	unsigned g;
	for(i = 0; i< N(); i++) {
		d = Vector3F(samples[i], groupCenter(i)).length();
		g = group(i);
		if(gs[g] < d) gs[g] = d;
	}
	/*
	for(i = 0; i < K(); i++)
		std::cout<<"\n count in group "<<i<<" "<<countPerGroup(i)
		<<" group centroid "<<centeroid(i)
		<<" size "<<m_groupSize[i];
	*/
	setValid(1);
}

void KMeanSampleGroup::setK(const unsigned & k)
{
	KMeansClustering::setK(k);
	createGroupSize(k);
}
