/*
 *  PoseSpaceDeformer.h
 *  eulerRot
 *
 *  Created by jian zhang on 11/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <SkeletonSubspaceDeformer.h>

class PoseSpaceDeformer : public SkeletonSubspaceDeformer {
public:
	struct PoseDelta {
		PoseDelta(unsigned idx, unsigned n) {
			_poseIdx = idx;
			_delta = new Vector3F[n];
		}
		~PoseDelta() {
			delete[] _delta;
		}
		unsigned _poseIdx;
		Vector3F * _delta;
	};
	
	PoseSpaceDeformer();
	virtual ~PoseSpaceDeformer();
	virtual void bindToSkeleton(SkeletonSystem * skeleton);
	
	void addPose(unsigned idx);
	void selectPose(unsigned idx);
	void updatePose(unsigned idx);
protected:
	virtual Vector3F bindP(unsigned idx, unsigned j) const;
private:
	PoseDelta * findPose(unsigned idx);
	std::vector<PoseDelta *> m_poses;
	Vector3F * m_delta;
};