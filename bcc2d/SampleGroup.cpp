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
}

SampleGroup::~SampleGroup() 
{
	if(m_groupSize) delete[] m_groupSize;
}

void SampleGroup::compute(Vector3F * samples, unsigned numSamples, unsigned numGroups)
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
	
	for(i = 0; i < K(); i++) m_groupSize[i] = 0.f;
	
	unsigned g;
	for(i = 0; i< N(); i++) {
		d = Vector3F(samples[i], groupCenter(i)).length();
		g = group(i);
		if(m_groupSize[g] < d) m_groupSize[g] = d;
	}
	/*
	for(i = 0; i < K(); i++)
		std::cout<<"\n count in group "<<i<<" "<<countPerGroup(i)
		<<" group centroid "<<centeroid(i)
		<<" size "<<m_groupSize[i];
	*/
	setValid(1);
}

void SampleGroup::setK(const unsigned & k)
{
	KMeansClustering::setK(k);
	if(m_groupSize) delete[] m_groupSize;
	m_groupSize = new float[k];
}
