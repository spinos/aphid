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
class SkeletonPose;
class PoseSpaceDeformer : public SkeletonSubspaceDeformer {
public:
	struct PoseDelta {
		PoseDelta(SkeletonPose * p) {
			_pose = p;
			_delta = new Vector3F[p->dof()];
		}
		~PoseDelta() {
			delete _pose;
			delete[] _delta;
		}
		SkeletonPose * _pose;
		Vector3F * _delta;
	};
	
	PoseSpaceDeformer();
	virtual ~PoseSpaceDeformer();
	virtual void bindToSkeleton(SkeletonSystem * skeleton);
	
	void addPose(unsigned idx);
	//void selectPose(unsigned idx);
	void updatePose(unsigned idx);
	
	void addPose();
	void selectPose(const std::string & name);
	void updatePose();
	void recoverPose();
	void renamePose(const std::string & fromName, const std::string & toName);
	
	unsigned numPoses() const;
	SkeletonPose * pose(unsigned idx) const;
	SkeletonPose * currentPose() const;
	
protected:
	virtual Vector3F bindP(unsigned idx, unsigned j) const;
private:
	unsigned maxPoseIndex() const;
	PoseDelta * findPose(unsigned idx);
	std::vector<PoseDelta *> m_poses;
	SkeletonPose * m_activePose;
	Vector3F * m_delta;
};