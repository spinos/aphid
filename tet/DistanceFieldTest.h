/*
 *  DistanceFieldTest.h
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
#include <DrawDistanceField.h>

namespace ttg {

class DistanceFieldTest : public Scene, public aphid::DrawDistanceField {

	TetraField m_fld;
	aphid::BDistanceFunction m_distFunc;
	
public:
	DistanceFieldTest();
	virtual ~DistanceFieldTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);

private:
	void drawGraph(aphid::GeoDrawer * dr);
};

}