/*
 *  SelectCondition.h
 *  aphid
 *
 *  Created by jian zhang on 11/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
class SelectCondition {
public:	
	SelectCondition();
	
	void setDistanceFilter(char on);
	void setRegionFilter(char on);
	void setFacingFilter(char on);
	
	char byDistance() const;
	char byRegion() const;
	char byFacing() const;
	
	char filteredByFacing(const Vector3F & nor) const;
	char filteredByDistance(const Vector3F & pos) const;
	char filteredByProbability() const;
	
	void setProbability(float p);
	void setCenter(Vector3F c);
	void setNormal(Vector3F n);
	void setMaxDistance(float d);
	
	Vector3F center() const;
	Vector3F normal() const;
	float maxDistance() const;

private:
	Vector3F m_center;
	Vector3F m_normal;
	float m_maxDistance, m_probability;
	bool m_byDistance, m_byRegion, m_byFacing;
};