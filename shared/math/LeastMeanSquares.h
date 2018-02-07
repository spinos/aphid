/*
 *  LeastMeanSquares.h
 *  
 *
 *  Created by jian zhang on 2/8/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MATH_LEAST_MEAN_SQUARES_H
#define APH_MATH_LEAST_MEAN_SQUARES_H

namespace aphid {

/// p is number of filter taps

template<typename T, int P>
class LeastMeanSquares {

/// p recent observed signals at n [x(n),x(n-1),...,x(n-p+1)]^T
	T* m_x;
/// filter coefficients [h_0,...,h_p-1]^T
	T* m_h;
	
public:
	
	LeastMeanSquares();
	
	void setData(T* x, T* h);
/// draw signal x(n)
/// estimate h(n)
/// return y(n) <- h^T(n) . x(n)
	T predict(const T& x_n, const int& n);
	
private:
/// h^T(n) . x(n)
	T hdx() const;

};

template<typename T, int P>
LeastMeanSquares<T, P>::LeastMeanSquares()
{}

template<typename T, int P>
void LeastMeanSquares<T, P>::setData(T* x, T* h)
{
	m_x = x;
	m_h = h;
}

template<typename T, int P>
T LeastMeanSquares<T, P>::predict(const T& x_n, const int& n)
{
	if(n<1) {
		for(int i=0;i<P;++i) {
			m_x[i] = x_n;
		}
	}
	
	for(int i=P-1;i>0;--i) {
		m_x[i] = m_x[i-1];
	}
	m_x[0] = x_n;
	
/// desired signal by convolution
	T d_n = 0.0;
	T hh = .5;
	for(int i=0;i<P;++i) {
		d_n += m_x[i] * hh;
		hh *= 0.5;
	}
	
/// error of estimation	
	T e_n = d_n - hdx();
	
	T xdx = 0;
	for(int i=0;i<P;++i) {
		xdx += m_x[i] * m_x[i];
	}
	
/// update h(n)
	T mu = 2. / (T)P;
	for(int i=0;i<P;++i) {
		m_h[i] = m_h[i] + mu * e_n * m_x[i];
	}
	return hdx();
}

template<typename T, int P>
T LeastMeanSquares<T, P>::hdx() const
{
	T r = 0;
	for(int i=0;i<P;++i) {
		r += m_h[i] * m_x[i];
	}
	return r;
}

}

#endif
