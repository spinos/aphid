/*
 *  RodDeformer.cpp
 *  
 *
 *  Created by jian zhang on 8/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "RodDeformer.h"
#include <smp/UniformGrid8Sphere.h>

using namespace aphid;

RodDeformer::RodDeformer()
{}

RodDeformer::~RodDeformer()
{}

void RodDeformer::initStates()
{ 
	const int& np = particles()->numParticles();
	const int& ng = ghostParticles()->numParticles();
	m_states.reset(new float[3 * smp::UniformGrid8Sphere::sNumSamples * (np + ng) ]); 
}

void RodDeformer::saveState(const int& i)
{
	const int& np = particles()->numParticles();
	const int& ng = ghostParticles()->numParticles();
	const int hd = 3 * i * (np + ng);
	float* pp = &m_states[hd];
	memcpy(pp, particles()->pos(), np * 12);
	
	float* pg = &m_states[hd + 3 * np];
	memcpy(pg, ghostParticles()->pos(), ng * 12);
}

void RodDeformer::loadState(const int& i)
{
	const int& np = particles()->numParticles();
	const int& ng = ghostParticles()->numParticles();
	const int hd = 3 * i * (np + ng);
	float* pp = &m_states[hd];
	memcpy(particles()->pos(), pp, np * 12);
	
	float* pg = &m_states[hd + 3 * np];
	memcpy(ghostParticles()->pos(), pg, ng * 12);
}
