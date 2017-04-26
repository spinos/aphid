/*
 *  Vegetation.cpp
 *  
 *
 *  Created by jian zhang on 4/26/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Vegetation.h"
#include "VegetationPatch.h"
#include <cmath>

Vegetation::Vegetation() :
m_numPatches(1)
{
	for(int i=0;i<TOTAL_NUM_P;++i) {
		m_patches[i] = new VegetationPatch;
	}
	const float deltaAng = .8f / ((float)NUM_ANGLE - 1);
	for(int j=0;j<NUM_ANGLE;++j) {
		const float angj = deltaAng * j;
		for(int i=0;i<NUM_VARIA;++i) {
			m_patches[j * NUM_VARIA + i]->setTilt(angj);
		}
	}
}

Vegetation::~Vegetation()
{
	for(int i=0;i<TOTAL_NUM_P;++i) {
		delete m_patches[i];
	}
}

VegetationPatch * Vegetation::patch(const int & i)
{
	return m_patches[i];
}

void Vegetation::setNumPatches(int x)
{ m_numPatches = x; }

const int & Vegetation::numPatches() const
{ return m_numPatches; }

int Vegetation::getMaxNumPatches() const
{ return TOTAL_NUM_P; }

void Vegetation::rearrange()
{
	float px, pz = 0.f, py = 0.f, spacing;
	const float deltaAng = .8f / ((float)NUM_ANGLE - 1);
	for(int j=0;j<NUM_ANGLE;++j) {
		px = 0.f;
		for(int i=0;i<NUM_VARIA;++i) {
			const int k = j * NUM_VARIA + i;
			if(k >= m_numPatches) {
				return;
			}
			
			m_patches[k]->setTranslation(px, py, pz);
			
			spacing = m_patches[k]->yardRadius() * 2.f;
			px += spacing;
		}
		py += spacing * sin(deltaAng*j);
		pz -= spacing * cos(deltaAng*j);;
		
	}
}
