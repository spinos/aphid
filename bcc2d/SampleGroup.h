/*
 *  SampleGroup.h
 *  testbcc
 *
 *  Created by jian zhang on 6/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <KMeansClustering.h>

class SampleGroup {
public:
	SampleGroup();
	virtual ~SampleGroup();
	
	virtual void compute(Vector3F * samples, unsigned numSamples, unsigned numGroups);
    float * groupSize();
    Vector3F * groupCentroid();
    unsigned numGroups() const;
protected:
    void createGroupSize(unsigned n);
    void createGroupCentroid(unsigned n);
private:
	float * m_groupSize;
    Vector3F * m_groupCentroid;
    unsigned m_numGroups;
};

class KMeanSampleGroup : public SampleGroup, public KMeansClustering {
public:
	KMeanSampleGroup();
	virtual ~KMeanSampleGroup();
	
	virtual void compute(Vector3F * samples, unsigned numSamples, unsigned numGroups);
protected:
    virtual void setK(const unsigned & k);
	
private:
	
};