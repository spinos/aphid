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
	
public:
	Legendre2DTest();
	virtual ~Legendre2DTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:

};

}