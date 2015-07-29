#include "CudaConjugateGradientSolver.h"
#include <CUDABuffer.h>
#include <CudaReduction.h>
#include <CudaCSRMatrix.h>
#include <cuConjugateGradient_implement.h>
#include <AllMath.h>
#include <CudaDbgLog.h>
#include <boost/format.hpp>
#include <FemGlobal.h>

CudaConjugateGradientSolver::CudaConjugateGradientSolver()
{
    m_residual = new CUDABuffer;
	m_update = new CUDABuffer;
	m_prev = new CUDABuffer;
	m_d = new CUDABuffer;
	m_d2 = new CUDABuffer;
	m_rhs = new CUDABuffer;
	m_reduce = new CudaReduction;
}

CudaConjugateGradientSolver::~CudaConjugateGradientSolver()
{
    delete m_residual;
	delete m_update;
	delete m_prev;
	delete m_d;
	delete m_d2;
	delete m_rhs;
	delete m_reduce;
}

void CudaConjugateGradientSolver::setDimension(unsigned n)
{ m_dimension = n; }

void * CudaConjugateGradientSolver::residual()
{ return m_residual->bufferOnDevice(); }

void * CudaConjugateGradientSolver::previous()
{ return m_prev->bufferOnDevice(); }

void * CudaConjugateGradientSolver::updated()
{ return m_update->bufferOnDevice(); }

void * CudaConjugateGradientSolver::diff()
{ return m_d->bufferOnDevice(); }

void * CudaConjugateGradientSolver::diff2()
{ return m_d2->bufferOnDevice(); }

void CudaConjugateGradientSolver::initOnDevice()
{
    m_residual->create(m_dimension * 12);
    m_update->create(m_dimension * 12);
    m_prev->create(m_dimension * 12);
    m_d->create(m_dimension * 4);
    m_d2->create(m_dimension * 4);
    m_rhs->create(m_dimension * 12);
    m_reduce->initOnDevice();
}

void * CudaConjugateGradientSolver::rightHandSide()
{ return m_rhs->bufferOnDevice(); }

CUDABuffer * CudaConjugateGradientSolver::rightHandSideBuf()
{ return m_rhs; }

void CudaConjugateGradientSolver::solve(void * X, 
                CudaCSRMatrix * A,
                void * fixed,
                float * error)
{
	// cglg.writeVec3(m_rhs, m_dimension, "cg b", CudaDbgLog::FAlways);
	//cglg.writeMat33(A->valueBuf(), 
	//				A->numNonZero(), 
	//				" cg A ", CudaDbgLog::FAlways);
					
    cuConjugateGradient_prevresidual((float3 *)previous(),
                            (float3 *)residual(),
                            (mat33 *)A->deviceValue(),
                            (uint *)A->deviceRowPtr(),
                            (uint *)A->deviceColInd(),
                            (uint *)fixed,
                            (float3 *)X,
                            (float3 *)rightHandSide(),
                            m_dimension);
    
    for(int i=0;i<FemGlobal::CGSolverMaxNumIterations;i++) {
	    cuConjugateGradient_Ax((float3 *)previous(),
                            (float3 *)updated(),
                            (float3 *)residual(),
                            (float *)diff(),
                            (float *)diff2(),
                            (mat33 *)A->deviceValue(),
                            (uint *)A->deviceRowPtr(),
                            (uint *)A->deviceColInd(),
                            (uint *)fixed,
                            m_dimension);
        
        float d =0;
		float d2=0;
		
		m_reduce->sum<float>(d, (float *)diff(), m_dimension);
        m_reduce->sum<float>(d2, (float *)diff2(), m_dimension);
        
		if(fabs(d2)< 1e-10f)
			d2 = 1e-10f;

		float d3 = d/d2;
		cuConjugateGradient_addX((float3 *)X,
                            (float3 *)residual(),
                            (float *)diff(),
                            (float3 *)previous(),
                            (float3 *)updated(),
                            d3,
                            (uint *)fixed,
                            m_dimension);
        
        float d1 = 0.f;

        m_reduce->sum<float>(d1, (float *)diff(), m_dimension);
        
        if(error) *error = d1;
        
		if(d1 < 0.09f)
			break;

		if(fabs(d)<1e-10f)
			d = 1e-10f;

		float d4 = d1/d;
		cuConjugateGradient_addResidual((float3 *)previous(),
                            (float3 *)residual(),
                            d4,
                            (uint *)fixed,
                            m_dimension);             
	}
}

