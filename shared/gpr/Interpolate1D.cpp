/*
 *  Interpolate1D.cpp
 *  
 *
 *  Created by jian zhang on 12/17/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Interpolate1D.h"
#include <math/linearMath.h>
#include <gpr/RbfKernel.h>
#include <gpr/Covariance.h>

namespace aphid {
namespace gpr {

Interpolate1D::Interpolate1D() :
m_rbf(NULL)
{ 
	m_bound = Float2(-1.f, 1.f); 
	m_covTrain = new Covariance<float, RbfKernel<float> >();
	m_xTrain = new DenseMatrix<float >();
	m_yTrain = new DenseMatrix<float >();
	
}

Interpolate1D::~Interpolate1D()
{ 
	clearObservations(); 
	if(m_rbf) delete m_rbf;
	delete m_covTrain;
}

void Interpolate1D::clearObservations()
{ m_observations.clear(); }

void Interpolate1D::addObservation(const float & x,
					const float & y)
{ m_observations.push_back(Float2(x, y) ); }

void Interpolate1D::setObservation(const float & x,
					const float & y,
					const int & idx)
{
	if(idx >= numObservations() ) {
		addObservation(x, y);
		return;
	}
	
	m_observations[idx] = Float2(x, y);
}

int Interpolate1D::numObservations() const
{ return m_observations.size(); }

void Interpolate1D::setBound(const float & lft,
				const float & rgt)
{
	m_bound = Float2(lft, rgt);
}

bool Interpolate1D::learn()
{
	const int dim = numObservations();
	if(dim<2) {
		std::cout<<"Interpolate1D has too few observations "<<dim;
		return false;
	}
	
	m_xTrain->resize(dim, 1);
	m_yTrain->resize(dim, 1);
	
	for(int i=0;i<dim;++i) {
		m_xTrain->column(0)[i] = m_observations[i].x;
		m_yTrain->column(0)[i] = m_observations[i].y;
	}
	
	
	if(m_rbf) delete m_rbf;
	m_rbf = new RbfKernel<float> (.125f * (m_bound.y - m_bound.x) );
    
	return m_covTrain->create(*m_xTrain, *m_rbf);
}

float Interpolate1D::predict(const float & x) const
{
	DenseMatrix<float > xTest(1,1);
	xTest.column(0)[0] = x;
	Covariance<float, RbfKernel<float> > covTest;
    covTest.create(xTest, *m_xTrain, *m_rbf);
	
	DenseMatrix<float> KxKtraininv(covTest.K().numRows(),
									m_covTrain->Kinv().numCols() );
									
	covTest.K().mult(KxKtraininv, m_covTrain->Kinv() );
	
/// yPred = Ktest * inv(Ktrain) * yTrain
	DenseMatrix<float> yPred(1,1);
	KxKtraininv.mult(yPred, *m_yTrain);
	return yPred.column(0)[0];
}

}
}
