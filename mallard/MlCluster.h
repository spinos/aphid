/*
 *  MlCluster.h
 *  mallard
 *
 *  Created by jian zhang on 12/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <KMeansClustering.h>
class MlCalamus;
class MlCalamusArray;
class AccPatchMesh;
class CollisionRegion;
class MlCluster : public KMeansClustering {
public:
	MlCluster();
	virtual ~MlCluster();
	
	void compute(MlCalamusArray * calamus, AccPatchMesh * mesh, unsigned begin, unsigned end);

	Float2 * angles(unsigned idx) const;
	unsigned sampleIdx(unsigned idx) const;
	void recordAngles(MlCalamus * c, unsigned idx);
	void reuseAngles(MlCalamus * c, unsigned idx);
	
	short sampleNSeg(unsigned idx) const;
	float sampleBend(unsigned idx) const;
protected:
    virtual void setK(const unsigned & k);
	
private:
	void assignGroupSample(MlCalamusArray * calamus, AccPatchMesh * mesh, unsigned begin, unsigned grp);
	void createAngles(MlCalamusArray * calamus);
	unsigned * m_sampleIndices;
	short * m_sampleNSegs;
	unsigned * m_angleStart;
    float * m_sampleBend;
	Float2 * m_angles;
};