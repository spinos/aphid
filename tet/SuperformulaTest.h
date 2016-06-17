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

class SuperformulaTest : public Scene {

	aphid::Vector3F * m_X;
	int m_N;
	
public:
	SuperformulaTest();
	virtual ~SuperformulaTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual bool progressForward();
	virtual bool progressBackward();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	bool createSamples();
	aphid::Vector3F randomPnt(float a, float b, float n1, float n2, float n3, float n4) const;
	
};

}
#endif