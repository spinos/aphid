#ifndef TTG_KDISTANCETEST_H
#define TTG_KDISTANCETEST_H

/*
 *  KDistanceTest.h
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
#include "Container.h"

namespace ttg {

class KDistanceTest : public Scene, public aphid::DrawDistanceField {

	FieldTriangulation m_msh;
	aphid::BDistanceFunction m_distFunc;
	std::string m_fileName;
	Container<cvx::Triangle > m_container;
	
public:
	KDistanceTest(const std::string & filename);
	virtual ~KDistanceTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	virtual bool viewPerspective() const;
	
private:
	void drawTree(aphid::GeoDrawer * dr);
	
};

}
#endif        //  #ifndef KDISTANCETEST_H
