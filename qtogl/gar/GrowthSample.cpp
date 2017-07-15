/*
 *  GrowthSample.cpp
 *  garden
 *
 *  Created by jian zhang on 7/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GrowthSample.h"
#include <smp/EbpSphere.h>
#include <math/Plane.h>
#include <math/miscfuncs.h>
#include <smp/EbpMeshSample.h>
#include <geom/DiscMesh.h>

using namespace aphid;

GrowthSample::GrowthSample()
{}

GrowthSample::~GrowthSample()
{}

void GrowthSample::samplePot(const GrowthSampleProfile& prof)
{
	DiscMesh dsk(24);
	const int ndskv = dsk.numPoints();
	for(int i=0;i<ndskv;++i) {
		Vector3F vi = dsk.points()[i];
		vi.set(vi.x * 9.9f * prof.m_sizing, 
				RandomFn11(), 
				vi.y * 9.9f * prof.m_sizing);
		dsk.points()[i] = vi;
	}
	
	EbpMeshSample smsh;
	smsh.sample(&dsk);
	
	setAngle(prof.m_angle);
	setPortion(prof.m_portion);
	setNumSampleLimit(prof.m_numSampleLimit);
	
	processFilter<EbpMeshSample>(&smsh);
	
	const int& np = numFilteredSamples();
	const Vector3F* ps = filteredSamples();
	
	m_pnds.reset(new Ray[np]);
	
	for(int i=0;i<np;++i) {
		m_pnds[i].m_origin = ps[i];
		m_pnds[i].m_dir = Vector3F::YAxis;
	}
}

void GrowthSample::sampleBush(const GrowthSampleProfile& prof)
{
	const float relsz = prof.m_sizing * 1.1f;
	setAngle(prof.m_angle);
	setPortion(prof.m_portion);
	setNumSampleLimit(prof.m_numSampleLimit);
	
	EbpSphere sph;
	processFilter<EbpSphere>(&sph);
	
	const int& np = numFilteredSamples();
	const Vector3F* ps = filteredSamples();
	
	m_pnds.reset(new Ray[np]);
	
	const Vector3F ori(0.f, -10.f, 0.f);
	const Plane pln(0.f, -1.f, 0.f, 0.f);
	Vector3F rd;
	float rt;
	
	for(int i=0;i<np;++i) {
		Ray ri(ori, ps[i]);
		m_pnds[i] = ri;
		
		pln.rayIntersect(ri, rd, rt);
		
		m_pnds[i].m_origin = rd * relsz;
	}
	
	//std::cout<<"\n sph n sample "<<sph.numSamples()
	//	<<"\n fltd n sample "<<np;
	//std::cout.flush();
}

const int& GrowthSample::numGrowthSamples() const
{ return numFilteredSamples(); }

Matrix44F GrowthSample::getGrowSpace(const int& i,
					const GrowthSampleProfile& prof) const
{
	const float sc = 1.f - RandomF01() * .2f;
	Matrix44F tm;
	tm.scaleBy(sc);
	if(prof.m_tilt > 0.f) {
		tm.rotateX(-prof.m_tilt);
	}
	
	Vector3F modU(RandomFn11() * .14f, 0.f, RandomFn11() * .14f);
	modU += m_pnds[i].m_dir;
	modU.normalize();
	
	Vector3F modS = modU.perpendicular();
	Vector3F modF = modS.cross(modU);

	Matrix33F locM;
	locM.fill(modS, modU, modF);
	
	Quaternion rotQ(RandomFn11() * 0.5f * PIF, modU);
	Matrix33F locRot(rotQ);
	
	locM *= locRot;
	locM *= tm.rotation();
	tm.setRotation(locM);
	tm.setTranslation(m_pnds[i].m_origin);
	
	return tm;
}

const Vector3F& GrowthSample::growthPoint(const int& i) const
{ return m_pnds[i].m_origin; }

const Vector3F& GrowthSample::growthDirection(const int& i) const
{ return m_pnds[i].m_dir; }
	