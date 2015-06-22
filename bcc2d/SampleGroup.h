/*
 *  SampleGroup.h
 *  testbcc
 *
 *  Created by jian zhang on 6/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <KMeansClustering.h>
class SampleGroup : public KMeansClustering {
public:
	SampleGroup();
	virtual ~SampleGroup();
	
	void compute(Vector3F * samples, unsigned numSamples, unsigned numGroups);
protected:
    virtual void setK(const unsigned & k);
	
private:
	float * m_groupSize;
};