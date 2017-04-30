/*
 *  HeightBccGrid.cpp
 *  ttg
 *
 *  Created by jian zhang on 7/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "HeightBccGrid.h"

namespace aphid {

namespace ttg {

HeightSubdivCondition::HeightSubdivCondition() :
m_numSamples(5)
{}

void HeightSubdivCondition::setSigma(const float & x)
{ m_sigma = x * .7f; }

void HeightSubdivCondition::setSampleSize(const float & x)
{ m_sampleSize = x * .45f; }

bool HeightSubdivCondition::satisfied(const float & py) const
{ 
	if(py - m_sampleSize > m_height + m_deHeight.y) {
		return false;
	}
	
	if(py + m_sampleSize < m_height + m_deHeight.x) {
		return false;
	}
/// evenly divided 
	//return true;
	return m_deHeight.y - m_deHeight.x > m_sigma; 
}

HeightBccGrid::HeightBccGrid()
{}

HeightBccGrid::~HeightBccGrid()
{}

}
}

