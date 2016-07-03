/*
 *  ClosestSampleTest.h
 *  
 *
 *  Created by jian zhang on 7/3/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef TTG_CLOSESTSAMPLE_H
#define TTG_CLOSESTSAMPLE_H
#include <AllMath.h>

namespace ttg {

class ClosestSampleTest {

	aphid::Vector3F * m_smps;
	int m_N;
	
public:
	ClosestSampleTest(const std::vector<aphid::Vector3F> & src);
	virtual ~ClosestSampleTest();
	
	int getClosest(aphid::Vector3F & dst, float & d, 
				const aphid::Vector3F & toPnt) const;
				
	int getIntersect(aphid::Vector3F & dst, float & d, 
				const aphid::Vector3F & seg1,
				const aphid::Vector3F & seg2) const;
				
	int getClosestOnSegment(aphid::Vector3F & dst, float & d, 
				const aphid::Vector3F & seg1,
				const aphid::Vector3F & seg2) const;
				
private:
	aphid::Vector3F firstUp(const aphid::Vector3F & seg1,
				const aphid::Vector3F & seg2) const;
	
};

}
#endif