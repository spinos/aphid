/*
 *  Legendre3DTest.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Scene.h"
#include <AQuadMesh.h>

namespace ttg {

class Legendre3DTest : public Scene {

#define N_L3_DIM 3
#define N_L3_ORD 4
#define N_ORD3 64
#define N_L3_P 3
#define N_SEG 16
#define N_SEG3 4096
	float m_Wi[N_L3_ORD];
	float m_Xi[N_L3_ORD];
	float m_Pv[N_L3_ORD * (N_L3_P+1)];
	float m_Yijk[N_ORD3];
	float m_Coeijk[(N_L3_P+1)*(N_L3_P+1)*(N_L3_P+1)];
	aphid::Vector3F m_samples[N_SEG3];
	float m_exact[N_SEG3];
	float m_appro[N_SEG3];

public:
	Legendre3DTest();
	virtual ~Legendre3DTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	float exactMeasure(const float & x, const float & y, const float & z) const;
/// l-th x m-th y n-th z coefficient by integrate f(x,y,z)P(l,x)P(m,y)P(n,z)
	float computeCoeff(int l, int m, int n) const;
/// continuous function expressed as a linear combination of Legendre polynomials
	float approximate(const float & x, const float & y, const float & z) const;
	
	void drawSamples(const float * val, aphid::GeoDrawer * dr) const;
	
};

}