/*
 *  LinearRegression.h
 *  
 *  data for linear regression
 *  Y = b_0 + b_1 x_1 + b_2 x_2 + ... + b_k x_k + err
 *  approximation Y^hat = b_0 + b_1 x_1 + b_2 x_2 + ... + b_k x_k
 *  b_i is coefficeints beta
 *  x_i is independent variables by recent history y_i
 *  Y is predition base on past
 *  K is order of model
 *  estimator based on recursive least squares fitting algorithm
 *
 *  Created by jian zhang on 2/4/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MATH_LINEAR_REGRESSION_DATA_H
#define APH_MATH_LINEAR_REGRESSION_DATA_H

#include "linearMath.h"

namespace aphid {

template<typename T, int K>
class LinearRegressionData {

/// k recent history
	T m_y[K];
/// coefficient
	T m_b[K + 1];
/// inverse covariant matrix
	T m_P[(K+1)*(K+1)];
	
public:

	LinearRegressionData();
	
	void addSignal(const T& y_t, const int& t);
	
	float* b();
	float* y();
	float* P();
	
private:
/// b_i <- 0
/// x_i <- 0
/// P <- 0 + diagonal(1000)
	void clear();
/// discard first 
	void pushData(const T& y_t);
	
};

template<typename T, int K>
LinearRegressionData<T, K>::LinearRegressionData()
{}

template<typename T, int K>
void LinearRegressionData<T, K>::clear()
{
	memset(m_b, 0, (K + 1) * sizeof(T) );
	memset(m_y, 0, K * sizeof(T) );
	memset(m_P, 0, (K + 1) * (K + 1) * sizeof(T) );
	for(int i=0;i<K+1;++i) {
		m_P[(K+1)*i + i] = 1000;
	}
}

template<typename T, int K>
void LinearRegressionData<T, K>::addSignal(const T& y_t, const int& t)
{	
	if(t == 0) {
		clear();
	} 
	
	if(t >= K) {
		pushData(y_t);
	} else {
		m_y[t] = y_t;
	}
}

template<typename T, int K>
void LinearRegressionData<T, K>::pushData(const T& y_t)
{
	for(int i=0;i<K-1;++i) {
		m_y[i] = m_y[i+1];
	}
	m_y[K-1] = y_t;
}

template<typename T, int K>
float* LinearRegressionData<T, K>::b()
{ return m_b; }
	
template<typename T, int K>
float* LinearRegressionData<T, K>::y()
{ return m_y; }

template<typename T, int K>
float* LinearRegressionData<T, K>::P()
{ return m_P; }

template<typename T, int K>
class LinearRegressionPredictor {

	LinearRegressionData<T, K>* m_model;
	DenseVector<T> m_X;
	DenseMatrix<T> P;
	DenseVector<T> xtp;
	DenseVector<T> Q;
	DenseMatrix<T> qxt;
	DenseMatrix<T> qxtp;
	
public:

	LinearRegressionPredictor();
	
	void setData(LinearRegressionData<T, K>* model);
	
	T updateAndPredict(const T& Y_t, const int& t);
	
private:
/// x_i <- [1, y_i]
	void assembleX();
/// Y^hat <- b_i^T x_i	
	T predict();

};

template<typename T, int K>
LinearRegressionPredictor<T, K>::LinearRegressionPredictor()
{
	m_X.create(K+1);
	P.create(K+1, K+1);
	xtp.create(K + 1);
	Q.create(K + 1);
	qxt.create(K + 1,K + 1);
	qxtp.create(K + 1,K + 1);
	
}

template<typename T, int K>
void LinearRegressionPredictor<T, K>::setData(LinearRegressionData<T, K>* model)
{ m_model = model; }

template<typename T, int K>
T LinearRegressionPredictor<T, K>::updateAndPredict(const T& Y_t, const int& t)
{
	m_model->addSignal(Y_t, t);
	
/// update x
	assembleX(); 
	// std::cout<<"\n X"<<m_X;
	
/// prediction error using current b	
	const T et = Y_t - predict();
	
	P.copyData(m_model->P() );
	//std::cout<<"\n P"<<P;
	
	xtp.setZero();
	P.lefthandMult(xtp, m_X);
	
	const float lamda = .998f;
	float scal = 1.f / (lamda + xtp.dot(m_X) );
	Q.setZero();
	P.mult(Q, m_X);
	Q.scale(scal);
	//std::cout<<"\n Q"<<Q;
/// update b	
	DenseVector<T> B(m_model->b(), K+1);
	B.add(Q * et);
	//std::cout<<"\n b"<<B;
/// update P
	qxt.setZero();
	qxt.asVVt(Q, m_X);
		//std::cout<<"\n Qx^t "<<qxt;
	
	qxtp.setZero();
	qxt.mult(qxtp, P);
		//std::cout<<"\n Qx^tP "<<qxtp;
	
	P.minus(qxtp);
	P.scale(1.f/lamda);
	
	//std::cout<<"\n P"<<P;
	
	P.extractData(m_model->P() );
/// updated prediction
	return predict();
}

template<typename T, int K>
T LinearRegressionPredictor<T, K>::predict()
{
	DenseVector<T> B(m_model->b(), K+1);
	return B.dot(m_X);
}

template<typename T, int K>
void LinearRegressionPredictor<T, K>::assembleX()
{
/// just slope ?
	m_X[0] = 1;
	for(int i=0;i<K;++i) {
		m_X[i + 1] = 1.0 - 1.9 / (1.9 + (T)K);
	}
}

}
#endif