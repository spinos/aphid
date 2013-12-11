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
	
	float * angles(unsigned idx) const;
	unsigned sampleIdx(unsigned idx) const;
	void recordAngles(MlCalamus * c, unsigned idx);
	void reuseAngles(MlCalamus * c, unsigned idx);
	
	short sampleNSeg(unsigned idx) const;
	Vector3F sampleDir(unsigned idx) const;
	float sampleLength(unsigned idx) const;
protected:
    virtual void setK(unsigned k);
	
private:
	void createAngles(MlCalamusArray * calamus, unsigned begin);
	void computeSampleDirs(MlCalamusArray * calamus, AccPatchMesh * mesh);
    int * m_sampleIndices;
	Vector3F * m_sampleDirs;
	short * m_sampleNSegs;
	float * m_sampleLengths;
    unsigned * m_angleStart;
    float * m_angles;
};