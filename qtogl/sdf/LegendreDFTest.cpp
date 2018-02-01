/*
 *  LegendreDFTest.cpp
 *  sdf
 *  
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "LegendreDFTest.h"
#include <math/miscfuncs.h>
#include <math/Calculus.h>
#include <math/ANoise3.h>
#include <GeoDrawer.h>

using namespace aphid;

LegendreDFTest::LegendreDFTest() 
{}

LegendreDFTest::~LegendreDFTest() 
{}

bool LegendreDFTest::init()
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

float LegendreDFTest::computeCoeff(int l, int m, int n) const
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

float LegendreDFTest::approximate(const float & x, const float & y, const float & z) const
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

float LegendreDFTest::exactMeasure(const float & x, const float & y, const float & z) const
{
#if 0
	const Vector3F at(x, 1.03f, z);
	const Vector3F orp(-.5421f, -.7534f, -.386f);
	return y - ANoise3::Fbm((const float *)&at,
										(const float *)&orp,
										.7f,
										4,
										1.8f,
										.5f);
#else
	float cx = x * 1.1f + .1f;
	float cy = y * .9f + .3f;
	float cz = z * .7f + .8f;
	float r = sqrt(cx * cx + cy * cy + cz * cz);
	return r - 1.1f;
#endif
}

void LegendreDFTest::draw(GeoDrawer * dr)
{
	glPushMatrix();
	glScalef(8.f, 8.f, 8.f);
	
	glTranslatef(-3.f, 0.f, 0.f);
	drawSamples(m_exact, dr);
	
	glTranslatef(3.f, 0.f, 0.f);
	drawSamples(m_appro, dr);
	
	glPopMatrix();
	
	if(m_isIntersected) {
		glColor3f(0.f,1.f,.1f);
		glBegin(GL_LINES);
		glVertex3fv((const float* )&m_oriP);
		glVertex3fv((const float* )&m_hitP);
		glEnd();
		dr->arrow(m_hitP, m_hitP + m_hitN * 8.f);
	}
}

void LegendreDFTest::drawSamples(const float * val, GeoDrawer * dr) const
{
	const float ssz = 1.f / N_SEG;
	int i=0;
	for(;i<N_SEG3;++i) {
		const float & r = val[i];
		if(r > 0.f)
			continue;
		
		dr->setColor(0.f,0.f,1.f + r);
		dr->cube(m_samples[i], ssz);
	}
}

void LegendreDFTest::rayIntersect(const Ray* ray)
{
	m_isIntersected = false;
	BoundingBox bx(-8.f, -8.f, -8.f, 8.f, 8.f, 8.f);
	float tmin, tmax;
	if(!bx.intersect(*ray, &tmin, &tmax) )
		return;
		
	m_hitP = ray->travel(tmin);
	m_oriP = ray->m_origin;
	
	float fd = approximate(m_hitP.x * .125f, m_hitP.y * .125f, m_hitP.z * .125f);
	if(fd < 0.f)
		return;
		
	int step = 0;
	while(fd > 1e-3f) {
		m_hitP += ray->m_dir * (fd * 8.f);
		if(!bx.isPointInside(m_hitP) )
			return;
			
		fd = approximate(m_hitP.x * .125f, m_hitP.y * .125f, m_hitP.z * .125f);
		// std::cout<<"\n d "<<step<<" "<<fd;
		// std::cout.flush();
		step++;
		if(step > 19)
			break;
	}
	
	calculateNormal(m_hitN, fd, m_hitP.x * .125f, m_hitP.y * .125f, m_hitP.z * .125f);
	m_isIntersected = true;
}

void LegendreDFTest::calculateNormal(Vector3F& nml, const float& q, const float & x, const float & y, const float & z) const
{
	nml.x = approximate(x + 6.25e-3f, y, z) - q;
	nml.y = approximate(x, y + 6.25e-3f, z) - q;
	nml.z = approximate(x, y, z + 6.25e-3f) - q;
	nml.normalize();
}
