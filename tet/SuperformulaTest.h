/*
 *  SuperformulaTest.h
 *  
 *
 *  Created by jian zhang on 6/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_SUPERFORMULA_TEST_H
#define TTG_SUPERFORMULA_TEST_H
#include "Scene.h"

namespace ttg {

class SuperformulaBase {

	float m_a1, m_b1, m_m1, m_n1, m_n2, m_n3;
	float m_a2, m_b2, m_m2, m_n21, m_n22, m_n23;
	
public:
	SuperformulaBase();
	virtual ~SuperformulaBase();

	void setA1(double x);
	void setB1(double x);
	void setM1(double x);
	void setN1(double x);
	void setN2(double x);
	void setN3(double x);
	void setA2(double x);
	void setB2(double x);
	void setM2(double x);
	void setN21(double x);
	void setN22(double x);
	void setN23(double x);

protected:
	virtual bool createSamples();
	
	aphid::Vector3F randomPnt(float u, float v) const;
							
private:	
	aphid::Vector3F randomPnt(float u, float v, float a, float b, 
							float m, float n1, float n2, float n3,
							float a2, float b2,
							float m2, float n21, float n22, float n23) const;
							
};

class SuperformulaTest : public Scene, public SuperformulaBase {

	aphid::Vector3F * m_X;
	int m_N, m_NDraw;
	
public:
	SuperformulaTest();
	virtual ~SuperformulaTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual bool progressForward();
	virtual bool progressBackward();
	virtual void draw(aphid::GeoDrawer * dr);
	
protected:
	void setN(int x);
	void setNDraw(int x);
	aphid::Vector3F * X();
	virtual bool createSamples();
	
};

}
#endif