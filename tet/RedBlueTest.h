/*
 *  RedBlueTest.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <AGraph.h>
#include "Scene.h"
#include "RedBlueRefine.h"

namespace ttg {

class RedBlueTest : public Scene {

	RedBlueRefine m_rbr;
	float m_nodeDrawSize;
	aphid::Vector3F m_p[10];
	float m_d[4];
	
public:
	RedBlueTest();
	virtual ~RedBlueTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);

	void setA(double x);
	void setB(double x);
	void setC(double x);
	void setD(double x);
	
private:
	void doRefine();
	void checkTetraVolume();
	
};

}