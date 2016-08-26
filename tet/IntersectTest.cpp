/*
 *  IntersectTest.cpp
 *  foo
 *  
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "IntersectTest.h"
#include <GridTables.h>
#include <Calculus.h>
#include <iostream>
#include <iomanip>

using namespace aphid;
namespace ttg {

IntersectTest::IntersectTest() 
{}

IntersectTest::~IntersectTest() 
{}
	
const char * IntersectTest::titleStr() const
{ return "Ray Intersect"; }

bool IntersectTest::init()
{
	m_sphere.set(Vector3F(0.f, 0.f, 0.f), 10.f);
	m_rays[0] = Ray(Vector3F(-20.f, 0.f, 0.f),
					Vector3F(10.f, 2.f, 0.f) );
	m_rays[1] = Ray(Vector3F(-20.f, 0.f, 0.f),
					Vector3F(20.f, 10.f, 0.f) );
	m_rays[2] = Ray(Vector3F(-14.414f, 0.f, 0.f),
					Vector3F(0.f, -14.414f, 0.f) );
	m_rays[3] = Ray(Vector3F(-5.f, 0.f, 0.f),
					Vector3F(-10.f, -2.f, 3.f) );
	m_rays[4] = Ray(Vector3F(15.f, 5.f, 0.f),
					Vector3F(-10.f, -10.f, 2.f) );
	m_rays[5] = Ray(Vector3F(5.f, 5.f, 0.f),
					Vector3F(-10.f, -15.f, 1.f) );
	m_rays[6] = Ray(Vector3F(5.f, -5.f, 0.f),
					Vector3F(0.f, 15.f, -1.f) );				
	m_rays[7] = Ray(Vector3F(3.f, -3.f, 0.f),
					Vector3F(3.f, -9.f, -2.f) );
	m_rays[8] = Ray(Vector3F(10.1f, 1.f, 0.f),
					Vector3F(9.1f, 1.f, 0.f) );
	m_rays[9] = Ray(Vector3F(9.91f, -1.f, 0.f),
					Vector3F(10.91f, -2.f, 0.f) );
	return true;
}

void IntersectTest::draw(GeoDrawer * dr)
{
	dr->alignedCircle(m_sphere.center(), m_sphere.radius() );
	
	for(int i=0;i<N_NUM_RAY;++i) {
		dr->setColor(.2f, .2f, .2f);
		dr->arrow(m_rays[i].m_origin, m_rays[i].destination() );
		
		float d = m_sphere.rayIntersect(m_rays[i]);
		
		if(d<1.f) {
			dr->setColor(.2f, .92f, .2f);
			dr->arrow(m_rays[i].m_origin, m_rays[i].travel(d*m_rays[i].m_tmax) );
		}
		
		m_rays[i].m_dir.x += RandomFn11() * .013f;
		m_rays[i].m_dir.y += RandomFn11() * .013f;
		m_rays[i].m_dir.z += RandomFn11() * .013f;
		m_rays[i].m_dir.normalize();
		m_rays[i].m_tmax += RandomFn11() * .1f;
	}
}

}