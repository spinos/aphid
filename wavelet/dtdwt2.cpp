/*
 *  dtdwt2.cpp
 *  
 *  http://eeweb.poly.edu/iselesni/WaveletSoftware/dt2D.html
 *  Created by jian zhang on 9/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "dtdwt2.h"
#include "dwt2.h"
#include <iostream>

namespace aphid {

namespace wla {

DualTree2::DualTree2() :
m_lastStage(0),
m_numRanks(0)
{}

DualTree2::~DualTree2()
{}

void DualTree2::createStage(int j, int m, int n, int p)
{
	for(int i=0; i<6;++i) {
		m_w[j][i].create(m, n, p);
	}
}
	
void DualTree2::analize(const Array3<float> & x, const int & nstage)
{
	const int & m = x.numRows();
	const int & n = x.numCols();
	const int & p = x.numRanks();
	
	int i, j, k;
	Array3<float> xn(x);
	for(k=0;k<p;++k) {
		*xn.rank(k) *= 0.707106781187f;
	}
	
	m_lastStage = 0;

	Array3<float> xu;
	xu.create(m/2, n/2, p);
	Array3<float> xd;
	xd.create(m/2, n/2, p);
	
	createStage(0, m/2, n/2, p);

	j = 0;
	
	for(k=0;k<p;++k) {
		afb2flt(xn.rank(k), Dtf::FirstStageUpFarrasAnalysis,
				xu.rank(k), m_w[0][0].rank(k), m_w[0][1].rank(k), m_w[0][2].rank(k));
		afb2flt(xn.rank(k), Dtf::FirstStageDownFarrasAnalysis,
				xd.rank(k), m_w[0][3].rank(k), m_w[0][4].rank(k), m_w[0][5].rank(k));
	}
	
	j++;
	
	while(j<nstage && j<(DT2_MAX_N_STAGE-1) ) {
	
		const int mj = xu.numRows();
		const int nj = xu.numCols();
		createStage(j, mj, nj, p);
	
		for(k=0;k<p;++k) {
			afb2flt(xu.rank(k), Dtf::UpAnalysis,
					xu.rank(k), m_w[j][0].rank(k), m_w[j][1].rank(k), m_w[j][2].rank(k) );
			
			afb2flt(xd.rank(k), Dtf::DownAnalysis,
					xd.rank(k), m_w[j][3].rank(k), m_w[j][4].rank(k), m_w[j][5].rank(k) );
		}
		j++;
		
	}
	
	m_w[j][0] = xu;
	m_w[j][3] = xd;
	
	m_lastStage = j;
/*
/// http://eeweb.poly.edu/iselesni/WaveletSoftware/allcode/dualtree2D.m
/// sum and difference
	for(j=0;j<m_lastStage;++j) {
		for(i=0;i<3;++i) {
			for(k=0;k<p;++k) {
				Array2<float> apb = *m_w[j][0+i].rank(k) + *m_w[j][3+i].rank(k);
				Array2<float> amb = *m_w[j][3+i].rank(k) - *m_w[j][3+i].rank(k);
				*m_w[j][0+i].rank(k) = apb;
				*m_w[j][0+i].rank(k) *= 0.707106781187f;
				*m_w[j][3+i].rank(k) = amb;
				*m_w[j][3+i].rank(k) *= 0.707106781187f;
			}
		}
	}*/
}

void DualTree2::synthesize(Array3<float> & y)
{
	const int & p = m_w[m_lastStage][0].numRanks();
	int i, j, k;
/*
/// http://eeweb.poly.edu/iselesni/WaveletSoftware/allcode/idualtree2D.m
/// sum and difference
	for(j=0;j<m_lastStage;++j) {
		for(i=0;i<3;++i) {
			for(k=0;k<p;++k) {
				Array2<float> apb = *m_w[j][0+i].rank(k) + *m_w[j][3+i].rank(k);
				Array2<float> amb = *m_w[j][3+i].rank(k) - *m_w[j][3+i].rank(k);
				*m_w[j][0+i].rank(k) = apb;
				*m_w[j][0+i].rank(k) *= 0.707106781187f;
				*m_w[j][3+i].rank(k) = amb;
				*m_w[j][3+i].rank(k) *= 0.707106781187f;
			}
		}
	}*/
	
	j = m_lastStage;
	Array3<float> yu;
	yu = m_w[j][0];
	Array3<float> yd;
	yd = m_w[j][3];
	
	j--;
	while(j>0) {
		std::cout<<"\n synj"<<j;
		for(k=0;k<p;++k) {
			sfb2flt(yu.rank(k), Dtf::UpSynthesis,
					yu.rank(k), m_w[j][0].rank(k), m_w[j][1].rank(k), m_w[j][2].rank(k) );
			sfb2flt(yd.rank(k), Dtf::DownSynthesis,
					yd.rank(k), m_w[j][3].rank(k), m_w[j][4].rank(k), m_w[j][5].rank(k) );
		}
		
		j--;
	}
	std::cout<<"\n synj"<<j;
	for(k=0;k<p;++k) {
		sfb2flt(yu.rank(k), Dtf::FirstStageUpFarrasSynthesis,
				yu.rank(k), m_w[j][0].rank(k), m_w[j][1].rank(k), m_w[j][2].rank(k) );
		sfb2flt(yd.rank(k), Dtf::FirstStageUpFarrasSynthesis,
				yd.rank(k), m_w[j][3].rank(k), m_w[j][4].rank(k), m_w[j][5].rank(k) );
	}
	
	y = yu;
	for(k=0;k<p;++k) {
		*y.rank(k) += *yd.rank(k);
		*y.rank(k) *= 0.707106781187f;
	}
	
}
	
const int & DualTree2::lastStage() const
{ return m_lastStage; }

const Array3<float> & DualTree2::stage(const int & i, const int & j, const int & k ) const
{ return m_w[i][j*3+k]; }

}

}