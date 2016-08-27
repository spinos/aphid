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
					
	m_beams = Beam(Vector3F(11.f, 0.f, 0.f),
					Vector3F(30.f, 9.f, 0.f), 2.f, 4.f);
	
	m_triangleCenter[0].set(19.f, 0.f, 2.f);
	m_triangleCenter[1].set(31.f, 3.f, 3.f);
	m_triangleCenter[2].set(12.f, -1.f, 0.f);				
	
	return true;
}

void IntersectTest::draw(GeoDrawer * dr)
{
	dr->alignedCircle(m_sphere.center(), m_sphere.radius() );
	
	int i;
	for(i=0;i<N_NUM_RAY;++i) {
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
	
	dr->setColor(.2f, .2f, .2f);
	dr->arrow(m_beams.origin(), m_beams.destination() );
	m_beams.setLength(1.f + RandomF01(), 20.f + RandomF01() );
	dr->alignedCircle(m_beams.ray().travel(m_beams.tmin() ), m_beams.radiusAt(m_beams.tmin() ) );
	dr->alignedCircle(m_beams.destination(), m_beams.radiusAt(m_beams.tmax() ) );

	for(i=0;i<N_NUM_TRI;++i) {
		
		m_triangle[i].set(m_triangleCenter[i] + Vector3F(1.f, 1.f, 1.f) + Vector3F(2.f, 1.f, 1.f) * RandomFn11(),
				m_triangleCenter[i] + Vector3F(1.f, -1.f, 1.f) + Vector3F(2.f, 1.f, 1.f) * RandomFn11(),
				m_triangleCenter[i] + Vector3F(-1.f, -1.f, -1.f) + Vector3F(2.f, 1.f, 1.f) * RandomFn11() );
	}
	
	for(i=0;i<N_NUM_TRI;++i) {
		const cvx::Triangle & tri = m_triangle[i];
		
		glBegin(GL_LINES);
		dr->vertex(tri.P(0) );
		dr->vertex(tri.P(1) );
		dr->vertex(tri.P(1) );
		dr->vertex(tri.P(2) );
		dr->vertex(tri.P(2) );
		dr->vertex(tri.P(0) );
		glEnd();
	}
	
	cvx::Sphere bs;
	float t0, t1;
	for(i=0;i<N_NUM_TRI;++i) {
		const cvx::Triangle & tri = m_triangle[i];
		
		tri.getBoundingSphere(bs);
		dr->setColor(.2f, .2f, .2f);
		dr->alignedCircle(bs.center(), bs.radius() );
		
		if(tri.beamIntersect(m_beams, &t0, &t1) ) {
			dr->setColor(0.f, 1.f, 0.f);
			dr->alignedCircle(m_beams.ray().travel(t0), m_beams.radiusAt(t0) );
			dr->arrow(bs.center(), m_beams.ray().travel(t0) );
		}
	}

	
}

}