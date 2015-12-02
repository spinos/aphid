/*
 *  Psnr.h
 *  testla
 *
 *  Created by jian zhang on 11/29/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "linearMath.h"

namespace lfr {

template<typename T>
class Psnr {
/// Peak Signal-to-Noise Ratio
/// PSNR = 10 log10 ( R^2 / MSE )
/// Mean Square Error
/// MSE = Sum ( | y - y^ |^2 ) / n

	DenseVector<T> m_yhat;
	const DenseMatrix<T> * m_X;
	T m_sum;
	int m_n;
	
public:
	Psnr(const DenseMatrix<T> * X);
	virtual ~Psnr();
	
	void reset();
	void add(const DenseVector<T> & y, const DenseVector<T> & beta, const DenseVector<int> & ind);
	T finish() const;
	
protected:

private:

};

template<typename T>
Psnr<T>::Psnr(const DenseMatrix<T> * X)
{
	m_X = X;
	const int m = X->numRows();
	m_yhat.create(m);
}

template<typename T>
Psnr<T>::~Psnr() {}

template<typename T>
void Psnr<T>::reset()
{
	m_sum = T(0.0);
	m_n = 0;
}
	
template<typename T>
void Psnr<T>::add(const DenseVector<T> & y, const DenseVector<T> & beta, const DenseVector<int> & ind)
{
	const int k = m_X->numColumns();
	const int m = y.numElements();
	int nnz = 0;
	int i=0;
	for(;i<k;++i) {
		if(ind[i] < 0) break;
		nnz++;
	}
	
	if(nnz < 1) {
		m_n += m/3;
		return;
	}
	
	m_yhat.setZero();
	
	int j;
	for(i=0;i<m;i++) {
		for(j=0;j<nnz;j++) {
			m_yhat[i] += m_X->column(ind[j])[i] * beta[j];
		}
	}
	
/// to luma
	const int nl = m/3;
	for(i=0;i<nl;i++) {
#if 1
		T e = 0.299 * m_yhat[i] + 0.587 * m_yhat[i + nl] + 0.114 * m_yhat[i + nl*2];
		e -= 0.299 * y[i] + 0.587 * y[i + nl] + 0.114 * y[i + nl*2];
		m_sum += e * e;
#else
        T e = m_yhat[i] - y[i];
        m_sum += e * e;
        e = m_yhat[i+ nl] - y[i+ nl];
        m_sum += e * e;
        e = m_yhat[i+ nl*2] - y[i+ nl*2];
        m_sum += e * e;
#endif
	}
	m_n += nl;
}
	
template<typename T>
T Psnr<T>::finish() const
{ 
	// std::cout<<"\n sum "<<m_sum;
	return 10.0 * log10( T(1.0) / (1e-10 + m_sum / m_n) ); }
}