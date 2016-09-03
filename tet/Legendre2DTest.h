/*
 *  Legendre2DTest.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Scene.h"
#include <AQuadMesh.h>

namespace ttg {

class Legendre2DTest : public Scene {

	aphid::AQuadMesh m_exact;
	aphid::AQuadMesh m_appro;
#define N_DIM 2
#define N_ORD 4
#define N_ORD2 16
#define N_P 3
	float m_Wi[N_ORD];
	float m_Xi[N_ORD];
	float m_Pv[N_ORD * (N_P+1)];
	float m_Pii[N_P+1];
	float m_Yij[N_ORD2];
	float m_Coeij[(N_P+1)*(N_P+1)];

public:
	Legendre2DTest();
	virtual ~Legendre2DTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	float exactMeasure(const float & x, const float & y) const;
/// l-th x m-th y coefficient by integrate f(x,y)P(l,x)P(m,y)
	float computeCoeff(int l, int m) const;
/// continuous function expressed as a linear combination of Legendre polynomials
	float approximate(const float & x, const float & y) const;
	
};

}