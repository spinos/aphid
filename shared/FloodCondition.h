/*
 *  FloodCondition.h
 *  aphid
 *
 *  Created by jian zhang on 11/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <SelectCondition.h>
class BaseTexture;
class FloodCondition : public SelectCondition {
public:	
	FloodCondition();
	void setMinDistance(float d);
	const float & minDistance() const;
	float minDistance(const unsigned & faceIdx, const float & u, const float & v) const;

	void setDensityMap(BaseTexture * tex);
	
	void increaseNumSamples(const unsigned & faceIdx, unsigned & dst) const;
	void reduceScale(const unsigned & faceIdx, const float & u, const float & v, float & dst) const;
private:
	BaseTexture * m_density;
	float m_minDistance;
};