/*
 *  VDGridTest.h
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
#include "FieldTriangulation.h"
#include <DrawDistanceField.h>

namespace ttg {

class VDGridTest : public Scene, public aphid::DrawDistanceField {

	FieldTriangulation m_msh;
	aphid::BDistanceFunction m_distFunc;
	
public:
	VDGridTest();
	virtual ~VDGridTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	virtual bool viewPerspective() const;
	
private:
	void drawGraph(aphid::GeoDrawer * dr);
	void drawCut(aphid::GeoDrawer * dr);
	
};

}