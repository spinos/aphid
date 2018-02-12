/*
 *  LegendreInterpolation.h
 *  
 *  legendre polynomial interpolation
 *  P polynomial order D dimension order
 *  gaussian quadrature on [-1,1]
 *
 *  Created by jian zhang on 2/12/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MATH_LEGENDRE_INTERPOLATION_H
#define APH_MATH_LEGENDRE_INTERPOLATION_H

#include "Calculus.h"

namespace aphid {

template<typename T, int P, int D>
class LegendreInterpolation {

	T* m_fpp;
/// P to the power of D
	static int m_PpD;
	
public:
	
	LegendreInterpolation();
	~LegendreInterpolation();
	
/// measure Y and compute coefficients
	template<typename Tm>
	void compute3(T* ys, T* coeffs, Tm& measure);
/// continuous function expressed as a linear combination of Legendre polynomials
/// reconstruct at u
	static T Approximate3(const T* u, const T* coeffs);
/// (dq/dx,dq/dy,dq/dz) at u
	static void ApproximateGradient3(T* nml, const T& q, const T* u, const T* coeffs);
	
/// precompute w, x, pv
	static void Initialize();
	
	static T m_Wi[P];
	static T m_Xi[P];
	static T m_Pv[P*P];
	
protected:

private:
/// l-th x m-th y n-th z coefficient by integrate f(x,y,z)P(l,x)P(m,y)P(n,z)
	T computeCoeff3(const T* yijk, int l, int m, int n);
	
};

template<typename T, int P, int D>
T LegendreInterpolation<T, P, D>::m_Wi[P];

template<typename T, int P, int D>
T LegendreInterpolation<T, P, D>::m_Xi[P];

template<typename T, int P, int D>
T LegendreInterpolation<T, P, D>::m_Pv[P*P];

template<typename T, int P, int D>
int LegendreInterpolation<T, P, D>::m_PpD = 0;

template<typename T, int P, int D>
LegendreInterpolation<T, P, D>::LegendreInterpolation()
{ m_fpp = new T[m_PpD]; }

template<typename T, int P, int D>
LegendreInterpolation<T, P, D>::~LegendreInterpolation()
{ delete[] m_fpp; }

template<typename T, int P, int D>
void LegendreInterpolation<T, P, D>::Initialize()
{
	m_PpD = P;
	for(int i=1;i<D;++i) {
		m_PpD *= P;
	}
	
	calc::gaussQuadratureRule(P, m_Wi, m_Xi);
	std::cout<<"\n gauss quadrate rule of order "<<P;
	calc::printValues<float>("wi", P, m_Wi);
	calc::printValues<float>("xi", P, m_Xi);
	calc::legendreRule(P, P-1, m_Pv, m_Xi);
	calc::printValues<float>("poly", P * P, m_Pv);
	
}

template<typename T, int P, int D>
T LegendreInterpolation<T, P, D>::computeCoeff3(const T* yijk, int l, int m, int n)
{
	int indx[D];
	int i,j,k,il;
	
	for(k=0;k<P;++k) {
		indx[2] = k;
		for(j=0;j<P;++j) {
			indx[1] = j;
			for(i=0;i<P;++i) {
				indx[0] = i;
					
				il = calc::lexIndex(D, P, indx);
				
/// f(x,y,z)P(l,x)P(m,y)P(n,z)
				m_fpp[il] = yijk[il] * m_Pv[i+P*l] * m_Pv[j+P*m] * m_Pv[k+P*n];
			}
		}
	}
	
	T result = calc::gaussQuadratureRuleIntegrate(D, P, m_Xi, m_Wi, m_fpp);
	result /= calc::LegendrePolynomial::norm2(l)
				* calc::LegendrePolynomial::norm2(m)
				* calc::LegendrePolynomial::norm2(n);
	//std::cout<<"\n C("<<l<<","<<m<<","<<n<<") "<<result;
	return result;
}

template<typename T, int P, int D>
template<typename Tm>
void LegendreInterpolation<T, P, D>::compute3(T* ys, T* coeffs, Tm& measure)
{
	int i,j,k,l;
	int indx[D];
	int rnk = 0;
	for(;;) {
		calc::tuple_next(1, P, D, &rnk, indx);
		
		if(rnk==0)
			break;
	
		//calc::printValues<int>("tuple space", D, indx);
		//std::cout<<"\n measure at ("<<m_Xi[indx[0]-1]
		//		<<","<<m_Xi[indx[1]-1]
		//		<<","<<m_Xi[indx[2]-1]<<")";
				
		l = calc::lexIndex(D, P, indx, -1);
		
		ys[l] = measure.measureAt(m_Xi[indx[0]-1], 
						m_Xi[indx[1]-1], 
						m_Xi[indx[2]-1]);
		
	}
	
	for(k=0;k<P;++k) {
		indx[2] = k;
		for(j=0;j<P;++j) {
			indx[1] = j;
			for(i=0;i<P;++i) {
				indx[0] = i;
				
				l = calc::lexIndex(D, P, indx);
				coeffs[l] = computeCoeff3(ys, i, j, k);
			}
		}
	}
}

template<typename T, int P, int D>
T LegendreInterpolation<T, P, D>::Approximate3(const T* u, const T* coeffs)
{
	T result = 0;
	int indx[D];
	int i, j, k,l;
	for(k=0;k<P;++k) {
		indx[2] = k;
		for(j=0;j<P;++j) {
			indx[1] = j;
			for(i=0;i<P;++i) {
				indx[0] = i;
				
				l = calc::lexIndex(D, P, indx);
				result += coeffs[l] 
							* calc::LegendrePolynomial::P(i, u[0])
							* calc::LegendrePolynomial::P(j, u[1])
							* calc::LegendrePolynomial::P(k, u[2]);
			
			}
		}
	}
	return result;
}

template<typename T, int P, int D>
void LegendreInterpolation<T, P, D>::ApproximateGradient3(T* nml, const T& q, const T* u, const T* coeffs)
{
	T p[3];
	p[0] = u[0] + 6.25e-3f;
	p[1] = u[1];
	p[2] = u[2];
	nml[0] = Approximate3(p, coeffs) - q;
	p[0] = u[0];
	p[1] = u[1] + 6.25e-3f;
	p[2] = u[2];
	nml[1] = Approximate3(p, coeffs) - q;
	p[0] = u[0];
	p[1] = u[1];
	p[2] = u[2] + 6.25e-3f;
	nml[2] = Approximate3(p, coeffs) - q;
}

}

#endif