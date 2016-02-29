/*
 *  ActiveGroup.h
 *  btree
 *
 *  Created by jian zhang on 1/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Ray.h>
#include <Array.h>
#include <Dropoff.h>

namespace aphid {
namespace sdb {

class ActiveGroup {
	
public:
	enum VolumeType {
		VSphere,
		VCylinder
	};
	
	ActiveGroup();
	~ActiveGroup();
	void reset();
	int numSelected();
	void finish();
	void average(Array<int, VertexP> * d);
	void setDropoffFunction(Dropoff::DistanceFunction x);
	const Dropoff::DistanceFunction dropoffFunction() const;
	
	Array<int, VertexP> * vertices;
	float depthMin, threshold;
	Vector3F meanPosition, meanNormal;
	Ray incidentRay;
	
	void updateMinDepth(float d);
	float minDepth() const;
	
	void setSelectVolume(VolumeType t );
	bool isEmpty() const;
	
private:
	void calculateWeight(Array<int, VertexP> * d);
	Dropoff::DistanceFunction m_dropoffType;
	Dropoff *m_drop;
	VolumeType m_volumeType;
};
	
}
}