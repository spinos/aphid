/*
 *  ShapeMatchingRegion.cpp
 *  
 *
 *  Created by jian zhang on 1/13/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "ShapeMatchingRegion.h"
#include <math/MatrixC33F.h>

namespace aphid {

namespace pbd {

ShapeMatchingRegion::ShapeMatchingRegion() : m_numPoints(0), 
m_numEdges(0),
m_stiffness(.7f)
{}

void ShapeMatchingRegion::createRegion(const RegionVE& prof,
				const float* pos, const float* invMass)
{
	m_numPoints = prof._nv;
	for(int i=0;i<m_numPoints;++i) {
		m_ind[i] = prof._vinds[i];
	}

/// m_i	
	m_totalMass = 0.f;
	for(int i=0;i<m_numPoints;++i) {
		float im = invMass[ m_ind[i] ];
		if(im > 0)
			m_mass[i] = 1.f / im;
		else 
			m_mass[i] = 100000.f;
		m_totalMass += m_mass[i];
	}

	updateCenterOfMass(pos);

/// q_i <- p_i	
	for(int i=0;i<m_numPoints;++i) {
		m_q[0][i] = m_p[0][i];
		m_q[1][i] = m_p[1][i];
		m_q[2][i] = m_p[2][i];
	}
	
	updateGoalPosition();
	
	m_numEdges = prof._ne;
	for(int i=0;i<m_numEdges;++i) {
		m_edge[i][0] = prof._einds[i * 2];
		m_edge[i][1] = prof._einds[i * 2 + 1];
	}
	
	for(int i=0;i<m_numEdges;++i) {
		m_goalInd[i][0] = findGoalInd(m_edge[i][0]);
		m_goalInd[i][1] = findGoalInd(m_edge[i][1]);
	}
/// initial rotation	
	m_rotq.set(1.f, 0.f, 0.f, 0.f);
}

int ShapeMatchingRegion::findGoalInd(const int& x) const
{
	for(int i=0;i<m_numPoints;++i) {
		if(m_ind[i] == x)
			return i;
	}
	return 0;
}

void ShapeMatchingRegion::updateRegion(const float* pos)
{
	updateCenterOfMass(pos);
	updateGoalPosition();
}

const float* ShapeMatchingRegion::goalPosition(const int& i) const
{ return m_g[i]; }

void ShapeMatchingRegion::updateCenterOfMass(const float* pos)
{
/// x_i	
	for(int i=0;i<m_numPoints;++i) {
		int pj = m_ind[i];
		
		m_p[0][i] = pos[pj * 3];
		m_p[1][i] = pos[pj * 3 + 1];
		m_p[2][i] = pos[pj * 3 + 2];
	}
	
	memset(m_centerOfMass, 0, 12);
	
	for(int i=0;i<m_numPoints;++i) {
		m_centerOfMass[0] += m_p[0][i] * m_mass[i];
		m_centerOfMass[1] += m_p[1][i] * m_mass[i];
		m_centerOfMass[2] += m_p[2][i] * m_mass[i];
	}
	
	m_centerOfMass[0] /= m_totalMass;
	m_centerOfMass[1] /= m_totalMass;
	m_centerOfMass[2] /= m_totalMass;

/// p_i <- x_i - x_cm	
	for(int i=0;i<m_numPoints;++i) {
		m_p[0][i] -= m_centerOfMass[0];
		m_p[1][i] -= m_centerOfMass[1];
		m_p[2][i] -= m_centerOfMass[2];
	}
}

void ShapeMatchingRegion::updateGoalPosition()
{
	MatrixC33F A_pq;
	A_pq.setZero();
/// sigma (m_i p_i q_i^T)	
	for(int i=0;i<m_numPoints;++i) {
		
		A_pq.col(0)[0] += m_p[0][i] * m_q[0][i];
		A_pq.col(0)[1] += m_p[1][i] * m_q[0][i];
		A_pq.col(0)[2] += m_p[2][i] * m_q[0][i];
		
		A_pq.col(1)[0] += m_p[0][i] * m_q[1][i];
		A_pq.col(1)[1] += m_p[1][i] * m_q[1][i];
		A_pq.col(1)[2] += m_p[2][i] * m_q[1][i];
		
		A_pq.col(2)[0] += m_p[0][i] * m_q[2][i];
		A_pq.col(2)[1] += m_p[1][i] * m_q[2][i];
		A_pq.col(2)[2] += m_p[2][i] * m_q[2][i];
		
	}

	// std::cout<<"\n Apq "<<A_pq;
/// start from last rotation
	A_pq.extractRotation(m_rotq, 50);
	MatrixC33F R(m_rotq);
	
	// std::cout<<"\n R "<<R;
/// g_i <- R q_i + x_cm	
	for(int i=0;i<m_numPoints;++i) {
		m_g[i][0] = R.col(0)[0] * m_q[0][i]
					+ R.col(1)[0] * m_q[1][i]
					+ R.col(2)[0] * m_q[2][i]
					+ m_centerOfMass[0];
		m_g[i][1] = R.col(0)[1] * m_q[0][i]
					+ R.col(1)[1] * m_q[1][i]
					+ R.col(2)[1] * m_q[2][i]
					+ m_centerOfMass[1];
		m_g[i][2] = R.col(0)[2] * m_q[0][i]
					+ R.col(1)[2] * m_q[1][i]
					+ R.col(2)[2] * m_q[2][i]
					+ m_centerOfMass[2];
	}
	
}

void ShapeMatchingRegion::setStiffness(const float& x)
{ m_stiffness = x; }

const int& ShapeMatchingRegion::numEdges() const
{ return m_numEdges; }

const int& ShapeMatchingRegion::numPoints() const
{ return m_numPoints; }

const float& ShapeMatchingRegion::stiffness() const
{ return m_stiffness; }

void ShapeMatchingRegion::getEdge(int& v1, int& v2, const int& i) const
{ 
	v1 = m_edge[i][0];
	v2 = m_edge[i][1];
}

void ShapeMatchingRegion::getGoalInd(int& v1, int& v2, const int& i) const
{
	v1 = m_goalInd[i][0];
	v2 = m_goalInd[i][1];
}

void ShapeMatchingRegion::solvePositionConstraint(float* x, const float* q_n1, 
			const int& i, const int& j) const
{
	x[0] = q_n1[i * 3]     + (m_g[j][0] - q_n1[i * 3]    ) * m_stiffness;
	x[1] = q_n1[i * 3 + 1] + (m_g[j][1] - q_n1[i * 3 + 1]) * m_stiffness;
	x[2] = q_n1[i * 3 + 2] + (m_g[j][2] - q_n1[i * 3 + 2]) * m_stiffness;
}

}

}