/*
 *  JointPiece.h
 *  
 *
 *  Created by jian zhang on 11/4/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TOPO_JOINT_PIECE_H
#define APH_TOPO_JOINT_PIECE_H

#include <boost/scoped_array.hpp>

namespace aphid {

namespace topo {

struct JointData {
	float _posv[4];
	int _cls[4];
	JointData* _parent;
};

class JointPiece {
	
	boost::scoped_array<JointData > m_joints;
	int m_numJoints;
	
public:
	JointPiece();
	virtual ~JointPiece();
	
	void create(int n);
	void zeroJointPos();
/// add x to i-th joint
	void addJointPos(const float* x, const int& i);
	void averageJointPos();
	void zeroJointVal();
/// add x to i-th joint
	void addJointVal(const float& x, const int& i);
	void averageJointVal();
/// find ind to x-id joint, set x to ind
    void setJointInd(int& x) const;
	void connectJoints(bool reversed);
	void setParent(JointData* j);
	
	const int& numJoints() const;
	
	JointData* joints();
	const JointData* joints() const;
/// first joint has parent
	bool hasParent() const;
	
protected:

private:
};

}

}

#endif

