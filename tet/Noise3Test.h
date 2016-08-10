#ifndef TTG_NOISE3_TEST_H
#define TTG_NOISE3_TEST_H

/*
 *  Noise3Test.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Scene.h"

namespace ttg {

class Noise3Test : public Scene {

public:
	Noise3Test();
	virtual ~Noise3Test();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	
};

}
#endif        //  #ifndef Noise3Test_H
