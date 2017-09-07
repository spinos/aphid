/*
 *  FiberBundle.cpp
 *  
 *
 *  Created by jian zhang on 9/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FiberBundle.h"
#include <math/Ray.h>
#include <math/Matrix44F.h>

namespace aphid {

FiberBundleBuilder::FiberBundleBuilder()
{}

FiberBundleBuilder::~FiberBundleBuilder()
{ clear(); }

int FiberBundleBuilder::numStrands() const
{ return m_strands.size(); }

void FiberBundleBuilder::addStrand()
{ m_strands.push_back(new FiberBulder); }

void FiberBundleBuilder::addPoint(const Vector3F& pv)
{ lastStrand()->addPoint(pv); }

FiberBulder* FiberBundleBuilder::lastStrand()
{
	if(numStrands() < 1)
		addStrand();
		
	return m_strands.back();
}

void FiberBundleBuilder::clear()
{
	std::deque<FiberBulder* >::iterator it = m_strands.begin();
	for(;it != m_strands.end();++it) {
		delete *it;
	}
	m_strands.clear();
}

const FiberBulder* FiberBundleBuilder::strand(int i) const
{ return m_strands[i]; }

FiberBulder* FiberBundleBuilder::strand(int i)
{ return m_strands[i]; }

void FiberBundleBuilder::begin()
{ clear(); }

void FiberBundleBuilder::end()
{
	std::deque<FiberBulder* >::iterator it = m_strands.begin();
	for(;it != m_strands.end();++it) {
		(*it)->end();
	}
}

void FiberBundleBuilder::insertPoint(int i, int j,
			const Vector3F& pv)
{
	if(j >= numStrands() )
		return;
		
	m_strands[j]->insertPoint(i, pv);
}

FiberBundle::FiberBundle()
{}

FiberBundle::~FiberBundle()
{}

void FiberBundle::create(const FiberBundleBuilder& bld)
{
	const int n = bld.numStrands();
	
	m_numPnts = 0;
	m_numSegs = 0;
	for(int i=0;i<n;++i) {
		const FiberBulder* si = bld.strand(i);
		m_numPnts += si->numPoints();
		m_numSegs += si->numSegments();
	}
	
	m_pnts.reset(new OrientedPoint[m_numPnts]);
	m_segs.reset(new FiberUnit[m_numSegs]);
	m_strandBegins.reset(new int[n + 1]);
	m_fibers.reset(new Fiber[n]);
	m_numStrands = n;
	
	int strandB = 0, strandP = 0;
	for(int i=0;i<n;++i) {
		const FiberBulder* si = bld.strand(i);
		si->copyStrand(&m_pnts[strandP], &m_segs[strandB] );

		m_fibers[i].create(si, &m_pnts[strandP], &m_segs[strandB]);
		
		m_strandBegins[i] = strandP;
		strandP += si->numPoints();
		strandB += si->numSegments();
	}
	m_strandBegins[n] = strandP;
}

void FiberBundle::dump(FiberBundleBuilder* bld) const
{
	for(int i=0;i<m_numStrands;++i) {
		if(bld->numStrands() <= i) 
			bld->addStrand();
		
		FiberBulder* bi = bld->strand(i);
		strand(i)->dump(bi);
	}
}

void FiberBundle::initialize()
{
	for(int i=0;i<m_numStrands;++i)
		m_fibers[i].initialize();
}

void FiberBundle::update()
{
	for(int i=0;i<m_numStrands;++i)
		m_fibers[i].update();
}

const int& FiberBundle::numPoints() const
{ return m_numPnts; }

const int& FiberBundle::numSegments() const
{ return m_numSegs; }

const int& FiberBundle::numStrands() const
{ return m_numStrands; }

const Fiber* FiberBundle::strand(int i) const
{ return &m_fibers[i]; }

const OrientedPoint* FiberBundle::points() const
{ return m_pnts.get(); }

bool FiberBundle::selectPoint(float& minD, const Ray* incident)
{
	m_selectedPoint = m_selectedStrand = -1;
	
	int curStrand = 0;
	for(int i=0;i<m_numPnts;++i) {
		float d = incident->distanceToPoint(m_pnts[i]._x);
		if(minD > d) {
			minD = d;
			m_selectedPoint = i;
			m_selectedStrand = curStrand;
		}
		
		if(i + 1== m_strandBegins[curStrand + 1]) {
			curStrand++;
		}
	}
	return (m_selectedPoint > -1 
			&& m_selectedStrand > -1);
}

void FiberBundle::getSelectPointSpace(Matrix44F* tm) const
{
	if(m_selectedPoint < 0)
		return;
		
	tm->setRotation(m_pnts[m_selectedPoint]._q);
	tm->setTranslation(m_pnts[m_selectedPoint]._x);
}

void FiberBundle::moveSelectedPoint(const Vector3F& dv)
{
	if(m_selectedPoint < 0)
		return;
	m_pnts[m_selectedPoint]._x += dv;
}

void FiberBundle::rotateSelectedPoint(const Quaternion& dq)
{
	if(m_selectedPoint < 0)
		return;
	m_pnts[m_selectedPoint]._q = dq * m_pnts[m_selectedPoint]._q;
}

void FiberBundle::moveSelectedStrand(const Vector3F& dv)
{
	if(m_selectedPoint < 0)
		return;
		
	const int ve = m_strandBegins[m_selectedStrand + 1];
	for(int i=m_selectedPoint;i<ve;++i)
		m_pnts[i]._x += dv;
}

void FiberBundle::rotateSelectedStrand(const Quaternion& dq)
{
	if(m_selectedPoint < 0)
		return;
		
	const Matrix33F rotm(dq);
	const Vector3F& p0 = m_pnts[m_selectedPoint]._x;
	const int ve = m_strandBegins[m_selectedStrand + 1];
	for(int i=m_selectedPoint;i<ve;++i) {
		m_pnts[i]._q = dq * m_pnts[i]._q;
		if(i>m_selectedPoint) {
			Vector3F dp = m_pnts[i]._x - p0;
			dp = rotm.transform(dp);
			m_pnts[i]._x = p0 + dp;
		}
	}
}

}
