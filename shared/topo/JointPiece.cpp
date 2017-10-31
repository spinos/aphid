/*
 *  JointPiece.cpp
 *  
 *
 *  Created by jian zhang on 11/4/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "JointPiece.h"
#include <math/QuickSort.h>
#include <iostream>
namespace aphid {

namespace topo {

JointPiece::JointPiece() :
m_numJoints(0)
{}

JointPiece::~JointPiece()
{}

void JointPiece::create(int n)
{
	m_joints.reset(new JointData[n]);
	for(int i=0;i<n;++i) {
		m_joints[i]._cls[0] = i;
		m_joints[i]._parent = NULL;
	}
	m_numJoints = n;
}

void JointPiece::zeroJointPos()
{
	for(int i=0;i<m_numJoints;++i) {
		memset(m_joints[i]._posv, 0, 12);
		m_joints[i]._cls[1] = 0;
	}
}

void JointPiece::addJointPos(const float* x, const int& i)
{
	m_joints[i]._posv[0] += x[0];
	m_joints[i]._posv[1] += x[1];
	m_joints[i]._posv[2] += x[2];
	m_joints[i]._cls[1] += 1;
}

void JointPiece::averageJointPos()
{
	for(int i=0;i<m_numJoints;++i) {
		const float scal = 1.f / (float)m_joints[i]._cls[1];
		m_joints[i]._posv[0] *= scal; 
		m_joints[i]._posv[1] *= scal; 
		m_joints[i]._posv[2] *= scal; 
	}
}

void JointPiece::zeroJointVal()
{
	for(int i=0;i<m_numJoints;++i) {
		m_joints[i]._posv[3] = 0;
		m_joints[i]._cls[1] = 0;
	}
}

void JointPiece::addJointVal(const float& x, const int& i)
{
	m_joints[i]._posv[3] += x;
	m_joints[i]._cls[1] += 1;
}
	
void JointPiece::averageJointVal()
{
	for(int i=0;i<m_numJoints;++i) {
		const float scal = 1.f / (float)m_joints[i]._cls[1];
		m_joints[i]._posv[3] *= scal;
	}
}

void JointPiece::connectJoints()
{
typedef QuickSortPair<float, int> SortTyp;
	SortTyp* buf = new SortTyp[m_numJoints];
	
	for(int i=0;i<m_numJoints;++i) {
		buf[i].key = m_joints[i]._posv[3];
		buf[i].value = i;
	}
	
	QuickSort1::Sort<float, int>(buf, 0, m_numJoints - 1);
	
	JointData* tmp = new JointData[m_numJoints];
	memcpy(tmp, m_joints.get(), sizeof(JointData) * m_numJoints);
	
	for(int i=0;i<m_numJoints;++i) {
		m_joints[i] = tmp[buf[i].value];
	}
	
	delete[] buf;
	delete[] tmp;
	
	std::cout<<"\n j0 "<<m_joints[0]._posv[3];
	
	for(int i=1;i<m_numJoints;++i) {
		m_joints[i]._parent = &m_joints[i-1];
		
		std::cout<<" j"<<i<<" "<<m_joints[i]._posv[3];
	}
}

const int& JointPiece::numJoints() const
{ return m_numJoints; }

JointData* JointPiece::joints()
{ return m_joints.get(); }

const JointData* JointPiece::joints() const
{ return m_joints.get(); }

}

}
