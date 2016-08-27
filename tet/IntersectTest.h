/*
 *  IntersectTest.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Scene.h"
#include <ConvexShape.h>

namespace ttg {

class IntersectTest : public Scene {
	
#define N_NUM_RAY 10
#define N_NUM_TRI 3
	aphid::cvx::Sphere m_sphere;
	aphid::Ray m_rays[N_NUM_RAY];
	aphid::Beam m_beams;
	aphid::cvx::Triangle m_triangle[N_NUM_TRI];
	aphid::Vector3F m_triangleCenter[N_NUM_TRI];
	
public:
	IntersectTest();
	virtual ~IntersectTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
private:
	
};

}