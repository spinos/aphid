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
	
	Vector3F center;
	Vector3F normal;
	float maxDistance;
	bool m_byDistance, m_byRegion, m_byFacing;
};