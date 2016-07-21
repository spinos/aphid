/*
 *  AdaptiveGridTest.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <AGraph.h>
#include "Scene.h"
#include "TetraField.h"
#include <BDistanceFunction.h>
#include "AdaptiveBccGrid3.h"

namespace ttg {

class AdaptiveGridTest : public Scene {

	AdaptiveBccGrid3 m_grd;
	aphid::BDistanceFunction m_distFunc;
	float m_nodeDrawSize, m_nodeColScl;
	
public:
	AdaptiveGridTest();
	virtual ~AdaptiveGridTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);

private:
	void drawGrid(aphid::GeoDrawer * dr);
	void drawNode(BccCell3 * cell, aphid::GeoDrawer * dr);
	void subdivideGrid(int level);
	
};

}