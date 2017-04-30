/*
 *  ActiveRegion.h
 *  aphid
 *
 *  Created by jian zhang on 11/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <LineBuffer.h>

namespace aphid {

class ActiveRegion : public LineBuffer {
public:
	ActiveRegion();
	virtual ~ActiveRegion();
	
	unsigned numActiveRegionFaces() const;
	unsigned activeRegionFace(unsigned idx) const;
	char hasActiveRegion() const;
	void clearActiveRegion();
	void addActiveRegionFace(unsigned idx);
	
	virtual void resetActiveRegion();
	
private:
	std::vector<unsigned> m_regionFaces;
};

}