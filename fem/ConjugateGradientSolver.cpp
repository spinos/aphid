/*
 *  ConjugateGradientSolver.cpp
 *  fem
 *
 *  Created by jian zhang on 1/8/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ConjugateGradientSolver.h"
int i_max = 20;
ConjugateGradientSolver::ConjugateGradientSolver() {}
ConjugateGradientSolver::~ConjugateGradientSolver() {}
	
void ConjugateGradientSolver::init(unsigned n)
{
	m_b = new Vector3F[n];
	m_residual = new Vector3F[n];
	m_update = new Vector3F[n];
	m_prev = new Vector3F[n];
	m_A_row = new MatrixMap[n];
	m_IsFixed= new bool[n];
	m_numRows = n;
}

void ConjugateGradientSolver::solve(Vector3F * X) 
{	
    for(unsigned k=0;k<m_numRows;k++) {
		if(m_IsFixed[k])
			continue;
		m_residual[k] = m_b[k];
 
		MatrixMap::iterator Abegin = m_A_row[k].begin();
        MatrixMap::iterator Aend   = m_A_row[k].end();
		for (MatrixMap::iterator A = Abegin; A != Aend;++A)
		{
            unsigned j   = A->first;
			Matrix33F& A_ij  = A->second;
			//float v_jx = m_V[j].x;	
			//float v_jy = m_V[j].y;
			//float v_jz = m_V[j].z;
			Vector3F prod = A_ij * X[j];
			                // Vector3F(	A_ij[0][0] * v_jx+A_ij[0][1] * v_jy+A_ij[0][2] * v_jz, //A_ij * prev[j]
							//			A_ij[1][0] * v_jx+A_ij[1][1] * v_jy+A_ij[1][2] * v_jz,			
							//			A_ij[2][0] * v_jx+A_ij[2][1] * v_jy+A_ij[2][2] * v_jz);
			m_residual[k] -= prod;//  A_ij * v_j;
			
		}
		m_prev[k]= m_residual[k];
	}
	
	for(int i=0;i<i_max;i++) {
		float d =0;
		float d2=0;
		
	 	for(unsigned k=0;k<m_numRows;k++) {

			if(m_IsFixed[k])
				continue;

			m_update[k].setZero();
			 
			MatrixMap::iterator Abegin = m_A_row[k].begin();
			MatrixMap::iterator Aend   = m_A_row[k].end();
			for (MatrixMap::iterator A = Abegin; A != Aend;++A) {
				unsigned j   = A->first;
				Matrix33F& A_ij  = A->second;
				// float prevx = prev[j].x;
				// float prevy = prev[j].y;
				// float prevz = prev[j].z;
				Vector3F prod = A_ij * m_prev[j];
				// Vector3F(	A_ij[0][0] * prevx+A_ij[0][1] * prevy+A_ij[0][2] * prevz, //A_ij * prev[j]
									//		A_ij[1][0] * prevx+A_ij[1][1] * prevy+A_ij[1][2] * prevz,			
										//	A_ij[2][0] * prevx+A_ij[2][1] * prevy+A_ij[2][2] * prevz);
				m_update[k] += prod;//A_ij*prev[j];
				 
			}
			d += m_residual[k].dot(m_residual[k]);
			d2 += m_prev[k].dot(m_update[k]);
		} 
		
		if(fabs(d2)< 1e-10f)
			d2 = 1e-10f;

		float d3 = d/d2;
		float d1 = 0.f;

		
		for(unsigned k=0;k<m_numRows;k++) {
			if(m_IsFixed[k])
				continue;

			X[k] += m_prev[k]* d3;
			m_residual[k] -= m_update[k]*d3;
			d1 += m_residual[k].dot(m_residual[k]);
		}
		
		if(i >= i_max && d1 < 0.001f)
			break;

		if(fabs(d)<1e-10f)
			d = 1e-10f;

		float d4 = d1/d;
		
		for(unsigned k=0;k<m_numRows;k++) {
			if(m_IsFixed[k])
				continue;
			m_prev[k] = m_residual[k] + m_prev[k]*d4;
		}		
	}	
}

bool * ConjugateGradientSolver::isFixed() { return m_IsFixed; }
Vector3F * ConjugateGradientSolver::rightHandSide() { return m_b; }
Matrix33F * ConjugateGradientSolver::A(unsigned i, unsigned j) { return &m_A_row[i][j]; }
