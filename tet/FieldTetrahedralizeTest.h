/*
 *  FieldTetrahedralizeTest.h
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

namespace ttg {

class FieldTetrahedralizeTest : public Scene {

	TetraField m_fld;
	aphid::BDistanceFunction m_distFunc;
	float m_nodeDrawSize, m_nodeColScl;
	
public:
	FieldTetrahedralizeTest();
	virtual ~FieldTetrahedralizeTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);

private:
	void drawGraph(aphid::GeoDrawer * dr);
	void drawGrid(aphid::GeoDrawer * dr);
	
};

}