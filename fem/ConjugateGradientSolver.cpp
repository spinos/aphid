/*
 *  ConjugateGradientSolver.cpp
 *  fem
 *
 *  Created by jian zhang on 1/8/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ConjugateGradientSolver.h"
#include <CudaCSRMatrix.h>
#include <CUDABuffer.h>
#include <cuConjugateGradient_implement.h>
#include <CudaReduction.h>
#include <CudaDbgLog.h>
#include <boost/format.hpp>

CudaDbgLog ceglg("cgsolver.txt");

int i_max = 32;
ConjugateGradientSolver::ConjugateGradientSolver() {}
ConjugateGradientSolver::~ConjugateGradientSolver() 
{
    delete[] m_b;
    delete[] m_residual;
    delete[] m_update;
    delete m_prev;
	delete[] m_A_row;
	delete[] m_IsFixed;
	delete m_deviceIsFixed;
	delete m_deviceResidual;
	delete m_deviceUpdate;
	delete m_devicePrev;
	delete m_deviceD;
	delete m_deviceD2;
	delete m_deviceX;
	delete m_deviceRhs;
	delete m_reduce;
}
	
void ConjugateGradientSolver::init(unsigned n)
{
	m_b = new Vector3F[n];
	m_residual = new Vector3F[n];
	m_update = new Vector3F[n];
	m_prev = new BaseBuffer;
	m_A_row = new MatrixMap[n];
	m_IsFixed= new int[n];
	m_numRows = n;
	m_deviceIsFixed = new CUDABuffer;
	m_deviceResidual = new CUDABuffer;
	m_deviceUpdate = new CUDABuffer;
	m_devicePrev = new CUDABuffer;
	m_deviceD = new CUDABuffer;
	m_deviceD2 = new CUDABuffer;
	m_deviceX = new CUDABuffer;
	m_deviceRhs = new CUDABuffer;
	m_reduce = new CudaReduction;
	m_prev->create(m_numRows * 12);
}

void ConjugateGradientSolver::initOnDevice()
{
    m_deviceIsFixed->create(m_numRows * 4);
    m_deviceIsFixed->hostToDevice(m_IsFixed);
    
    m_deviceResidual->create(m_numRows * 12);
    m_deviceUpdate->create(m_numRows * 12);
    m_devicePrev->create(m_numRows * 12);
    m_deviceD->create(m_numRows * 4);
    m_deviceD2->create(m_numRows * 4);
    m_deviceX->create(m_numRows * 12);
    m_deviceRhs->create(m_numRows * 12);
    m_reduce->initOnDevice();
}

void ConjugateGradientSolver::solveGpu(Vector3F * X, CudaCSRMatrix * stiffnessMatrix) 
{	
    void * dFixed = m_deviceIsFixed->bufferOnDevice();
    m_deviceX->hostToDevice(X);
    void * dX = m_deviceX->bufferOnDevice();
    
    m_deviceRhs->hostToDevice(m_b);
    void * dRhs = m_deviceRhs->bufferOnDevice();
    
	cuConjugateGradient_prevresidual((float3 *)m_devicePrev->bufferOnDevice(),
                            (float3 *)m_deviceResidual->bufferOnDevice(),
                            (mat33 *)stiffnessMatrix->deviceValue(),
                            (uint *)stiffnessMatrix->deviceRowPtr(),
                            (uint *)stiffnessMatrix->deviceColInd(),
                            (uint *)dFixed,
                            (float3 *)dX,
                            (float3 *)dRhs,
                            m_numRows);
    
    // ceglg.write("cg init");
    // ceglg.writeVec3(m_deviceResidual, m_numRows, "cg residual", CudaDbgLog::FAlways);
    // ceglg.writeVec3(m_devicePrev, m_numRows, "cg prev", CudaDbgLog::FAlways);
    
    // m_devicePrev->deviceToHost(m_prev->data());

    Vector3F * prev = (Vector3F *)m_prev->data();
	
	for(unsigned k=0; k< 1; k++) {
        // std::cout<<" prev["<<k<<"] h "<<prev[k];
    }
    
	for(unsigned i=0;i<i_max;i++) {
	    cuConjugateGradient_Ax((float3 *)m_devicePrev->bufferOnDevice(),
                            (float3 *)m_deviceUpdate->bufferOnDevice(),
                            (float3 *)m_deviceResidual->bufferOnDevice(),
                            (float *)m_deviceD->bufferOnDevice(),
                            (float *)m_deviceD2->bufferOnDevice(),
                            (mat33 *)stiffnessMatrix->deviceValue(),
                            (uint *)stiffnessMatrix->deviceRowPtr(),
                            (uint *)stiffnessMatrix->deviceColInd(),
                            (uint *)dFixed,
                            m_numRows);
        
        // ceglg.write("cg step ");
        // ceglg.write(i);
    
        // ceglg.writeVec3(m_deviceUpdate, m_numRows, "cg update", CudaDbgLog::FAlways);
        // ceglg.writeVec3(m_deviceResidual, m_numRows, "cg residual", CudaDbgLog::FAlways);
        // ceglg.writeVec3(m_devicePrev, m_numRows, "cg prev", CudaDbgLog::FAlways);
    
        float d =0;
		float d2=0;
		
		m_reduce->sumF(d, (float *)m_deviceD->bufferOnDevice(), m_numRows);
        m_reduce->sumF(d2, (float *)m_deviceD2->bufferOnDevice(), m_numRows);
        
		if(fabs(d2)< 1e-10f)
			d2 = 1e-10f;

		float d3 = d/d2;
		cuConjugateGradient_addX((float3 *)dX,
                            (float3 *)m_deviceResidual->bufferOnDevice(),
                            (float *)m_deviceD->bufferOnDevice(),
                            (float3 *)m_devicePrev->bufferOnDevice(),
                            (float3 *)m_deviceUpdate->bufferOnDevice(),
                            d3,
                            (uint *)dFixed,
                            m_numRows);
        // ceglg.write("addX");
        // ceglg.writeVec3(m_deviceUpdate, m_numRows, "cg update", CudaDbgLog::FAlways);
        // ceglg.writeVec3(m_deviceResidual, m_numRows, "cg residual", CudaDbgLog::FAlways);
        // ceglg.writeVec3(m_devicePrev, m_numRows, "cg prev", CudaDbgLog::FAlways);
        // 
        float d1 = 0.f;

        m_reduce->sumF(d1, (float *)m_deviceD->bufferOnDevice(), m_numRows);
        
        // if(i>29) std::cout<<" d1["<<i<<"] "<<d1<<" ";
        
		if(i >= i_max && d1 < 0.001f)
			break;

		if(fabs(d)<1e-10f)
			d = 1e-10f;

		float d4 = d1/d;
		cuConjugateGradient_addResidual((float3 *)m_devicePrev->bufferOnDevice(),
                            (float3 *)m_deviceResidual->bufferOnDevice(),
                            d4,
                            (uint *)dFixed,
                            m_numRows);
        
        // ceglg.write("add residual");
        // ceglg.writeVec3(m_deviceUpdate, m_numRows, "cg update", CudaDbgLog::FAlways);
        // ceglg.writeVec3(m_deviceResidual, m_numRows, "cg residual", CudaDbgLog::FAlways);
        // ceglg.writeVec3(m_devicePrev, m_numRows, "cg prev", CudaDbgLog::FAlways);
        // 
	}	 
	cudaThreadSynchronize();
	m_deviceX->deviceToHost(X);
}

void ConjugateGradientSolver::solve(Vector3F * X) 
{	
    Vector3F * prev = (Vector3F *)m_prev->data();
    
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
		prev[k]= m_residual[k];
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
				Vector3F prod = A_ij * prev[j];
				// Vector3F(	A_ij[0][0] * prevx+A_ij[0][1] * prevy+A_ij[0][2] * prevz, //A_ij * prev[j]
									//		A_ij[1][0] * prevx+A_ij[1][1] * prevy+A_ij[1][2] * prevz,			
										//	A_ij[2][0] * prevx+A_ij[2][1] * prevy+A_ij[2][2] * prevz);
				m_update[k] += prod;//A_ij*prev[j];
				 
			}
			d += m_residual[k].dot(m_residual[k]);
			d2 += prev[k].dot(m_update[k]);
		} 
		
		if(fabs(d2)< 1e-10f)
			d2 = 1e-10f;

		float d3 = d/d2;
		float d1 = 0.f;

		
		for(unsigned k=0;k<m_numRows;k++) {
			if(m_IsFixed[k])
				continue;

			X[k] += prev[k]* d3;
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
			prev[k] = m_residual[k] + prev[k]*d4;
		}		
	}	
}

int * ConjugateGradientSolver::isFixed() { return m_IsFixed; }
Vector3F * ConjugateGradientSolver::rightHandSide() { return m_b; }
Matrix33F * ConjugateGradientSolver::A(unsigned i, unsigned j) { return &m_A_row[i][j]; }
