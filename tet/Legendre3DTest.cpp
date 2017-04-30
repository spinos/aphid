/*
 *  Legendre3DTest.cpp
 *  foo
 *  
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Legendre3DTest.h"
#include <GridTables.h>
#include <Calculus.h>
#include <ANoise3.h>

using namespace aphid;
namespace ttg {

Legendre3DTest::Legendre3DTest() 
{}

Legendre3DTest::~Legendre3DTest() 
{}
	
const char * Legendre3DTest::titleStr() const
{ return "3D Legendre Polynomial Approximation"; }

bool Legendre3DTest::init()
{
	int i,j,k,l;
	int indx[N_L3_DIM];
	
	const float du = 2.f / N_SEG;
	
	for(k=0;k<N_SEG;++k) {
		indx[2] = k;
		for(j=0;j<N_SEG;++j) {
			indx[1] = j;
			for(i=0;i<N_SEG;++i) {
				indx[0] = i;
				
				l = calc::lexIndex(N_L3_DIM, N_SEG, indx);
				
				m_samples[l].set(-1.f + du * (.5f + i),
								-1.f + du * (.5f + j),
								-1.f + du * (.5f + k) );
				
			}
		}
	}

	calc::gaussQuadratureRule(N_L3_ORD, m_Wi, m_Xi);
	std::cout<<"\n gauss quadrate rule of order "<<N_L3_ORD;
	calc::printValues<float>("wi", N_L3_ORD, m_Wi);
	calc::printValues<float>("xi", N_L3_ORD, m_Xi);
	calc::legendreRule(N_L3_ORD, N_L3_P, m_Pv, m_Xi);
	calc::printValues<float>("poly", N_L3_ORD * (N_L3_P+1), m_Pv);
	
	int rnk = 0;
	int neval = 0;
	for(;;) {
		calc::tuple_next(1, N_L3_ORD, N_L3_DIM, &rnk, indx);
		
		if(rnk==0)
			break;
	
		calc::printValues<int>("tuple space", N_L3_DIM, indx);
		std::cout<<"\n measure at ("<<m_Xi[indx[0]-1]<<","<<m_Xi[indx[1]-1]<<","<<m_Xi[indx[2]-1]<<")";
		l = calc::lexIndex(N_L3_DIM, N_L3_ORD, indx, -1);
		m_Yijk[l] = exactMeasure(m_Xi[indx[0]-1], m_Xi[indx[1]-1], m_Xi[indx[2]-1] );
		
		neval++;
	}
	
	std::cout<<"\n n eval "<<neval;
	
	for(k=0;k<=N_L3_P;++k) {
		indx[2] = k;
		for(j=0;j<=N_L3_P;++j) {
			indx[1] = j;
			for(i=0;i<=N_L3_P;++i) {
				indx[0] = i;
				
				l = calc::lexIndex(N_L3_DIM, N_L3_P+1, indx);
				m_Coeijk[l] = computeCoeff(i, j, k);
			}
		}
	}
	
	float err, mxErr = 0.f, sumErr = 0.f;
	for(k=0;k<N_SEG;++k) {
		indx[2] = k;
		for(j=0;j<N_SEG;++j) {
			indx[1] = j;
			for(i=0;i<N_SEG;++i) {
				indx[0] = i;
				
				l = calc::lexIndex(N_L3_DIM, N_SEG, indx);
				
				m_exact[l] = exactMeasure(m_samples[l].x, m_samples[l].y, m_samples[l].z);
				m_appro[l] = approximate(m_samples[l].x, m_samples[l].y, m_samples[l].z);
				
				err = Absolute<float>(m_appro[l] - m_exact[l]);
				if(mxErr < err)
					mxErr = err;
				sumErr += err;
			}
		}
	}
	
	std::cout<<"\n max error "<<mxErr<<" average "<<(sumErr/N_SEG3)
		<<"\n done!";
	std::cout.flush();
	return true;
}

float Legendre3DTest::computeCoeff(int l, int m, int n) const
{
	int indx[N_L3_DIM];
	float fpp[N_ORD3];
	int i,j,k,il;
	
	for(k=0;k<N_L3_ORD;++k) {
		indx[2] = k;
		for(j=0;j<N_L3_ORD;++j) {
			indx[1] = j;
			for(i=0;i<N_L3_ORD;++i) {
				indx[0] = i;
					
				il = calc::lexIndex(N_L3_DIM, N_L3_ORD, indx);
				
/// f(x,y,z)P(l,x)P(m,y)P(n,z)
				fpp[il] = m_Yijk[il] * m_Pv[i+N_L3_ORD*l] * m_Pv[j+N_L3_ORD*m] * m_Pv[k+N_L3_ORD*n];
			}
		}
	}
	
	float result = calc::gaussQuadratureRuleIntegrate(N_L3_DIM, N_L3_ORD,
												m_Xi, m_Wi, fpp);
	result /= calc::LegendrePolynomial::norm2(l)
				* calc::LegendrePolynomial::norm2(m)
				* calc::LegendrePolynomial::norm2(n);
	std::cout<<"\n C("<<l<<","<<m<<","<<n<<") "<<result;
	return result;
}

float Legendre3DTest::approximate(const float & x, const float & y, const float & z) const
{
#define U_P 3
	float result = 0.f;
	int indx[N_L3_DIM];
	int i, j, k,l;
	for(k=0;k<=U_P;++k) {
		indx[2] = k;
		for(j=0;j<=U_P;++j) {
			indx[1] = j;
			for(i=0;i<=U_P;++i) {
				indx[0] = i;
				
				l = calc::lexIndex(N_L3_DIM, N_L3_P+1, indx);
				result += m_Coeijk[l] 
							* calc::LegendrePolynomial::P(i, x)
							* calc::LegendrePolynomial::P(j, y)
							* calc::LegendrePolynomial::P(k, z);
			
			}
		}
	}
	return result;
}

float Legendre3DTest::exactMeasure(const float & x, const float & y, const float & z) const
{
	const Vector3F at(x, y, z);
	const Vector3F orp(.6241f, .8534f, .2786f);
	return ANoise3::Fbm((const float *)&at,
										(const float *)&orp,
										.33f,
										4,
										1.33f,
										.695f);
}

void Legendre3DTest::draw(GeoDrawer * dr)
{
	glPushMatrix();
	glScalef(10.f, 10.f, 10.f);
	
	glTranslatef(-1.4f, 0.f, 0.f);
	drawSamples(m_exact, dr);
	
	glTranslatef(2.8f, 0.f, 0.f);
	drawSamples(m_appro, dr);
	
	glPopMatrix();
}

void Legendre3DTest::drawSamples(const float * val, GeoDrawer * dr) const
{
	const float ssz = .8f / N_SEG;
	int i=0;
	for(;i<N_SEG3;++i) {
		const float & r = val[i];
		dr->setColor(r,r,r);
		dr->cube(m_samples[i], ssz);
	}
}

}