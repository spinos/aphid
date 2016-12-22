/*
 *  GPInterpolate.h
 *  
 *	multi-variable interpolation via gaussian process
 *
 *  Created by jian zhang on 12/23/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_GPR_GP_INTERPOLATE_H
#define APH_GPR_GP_INTERPOLATE_H

#include <gpr/RbfKernel.h>
#include <gpr/Covariance.h>

namespace aphid {

namespace gpr {

template<typename T>
class GPInterpolate {

/// nobs-by-nvar nobs nvar-dimensional data points stored rowwise
	DenseMatrix<T > m_xTrain;
	DenseMatrix<T > m_yTrain;
/// 1-by-nvar nvar-dimensional row vector
	DenseMatrix<T > m_xPredict;
	DenseMatrix<T > m_yPredict;
	RbfKernel<T > m_rbf;
	Covariance<T, RbfKernel<T > > m_covTrain;
	Covariance<T, RbfKernel<T > > m_covPredict;
	
public:
	GPInterpolate();
	virtual ~GPInterpolate();
	
	void create(const int & nobs,
				const int & xnvar,
				const int & ynvar);
	
	const int & numObservations() const;
	const int & numXVars() const;
	const int & numYVars() const;
	
/// set idx-th row of X and Y
	void setObservationi(const int & idx,
			const T * x, const T * y);
	
	bool learn();
	
	void predict(const T * x);
	
	const DenseMatrix<T > & predictedY() const;
	
protected:

private:

};

template<typename T>
GPInterpolate<T>::GPInterpolate()
{}

template<typename T>
GPInterpolate<T>::~GPInterpolate()
{}

template<typename T>
void GPInterpolate<T>::create(const int & nobs,
				const int & xnvar,
				const int & ynvar)
{
	m_xTrain.resize(nobs, xnvar);
	m_yTrain.resize(nobs, ynvar);
	m_xPredict.resize(1, xnvar);
	m_yPredict.resize(1, ynvar);
	m_rbf.setParameter(3.0, 1.0);
	
}

template<typename T>
const int & GPInterpolate<T>::numObservations() const
{ return m_xTrain.numRows(); }

template<typename T>
const int & GPInterpolate<T>::numXVars() const
{ return m_xTrain.numCols(); }

template<typename T>
const int & GPInterpolate<T>::numYVars() const
{ return m_yTrain.numCols(); }

template<typename T>
void GPInterpolate<T>::setObservationi(const int & idx,
			const T * x, const T * y)
{ 
	m_xTrain.copyRow(idx, x); 
	m_yTrain.copyRow(idx, y); 
}

template<typename T>
bool GPInterpolate<T>::learn()
{
	return m_covTrain.create(m_xTrain, m_rbf);
}

template<typename T>
void GPInterpolate<T>::predict(const T * x)
{
	m_xPredict.copyRow(0, x);
	m_covPredict.create(m_xPredict, m_xTrain, m_rbf);
	
	DenseMatrix<float> KxKtraininv(m_covPredict.K().numRows(),
									m_covTrain.Kinv().numCols() );
									
	m_covPredict.K().mult(KxKtraininv, m_covTrain.Kinv() );
	
/// yPred = Ktest * inv(Ktrain) * yTrain
	KxKtraininv.mult(m_yPredict, m_yTrain);
}

template<typename T>
const DenseMatrix<T > & GPInterpolate<T>::predictedY() const
{ return m_yPredict; }

}

}
#endif
