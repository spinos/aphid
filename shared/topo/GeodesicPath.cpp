/*
 *  GeodesicPath.cpp
 * 
 *  Created by jian zhang on 10/26/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GeodesicPath.h"
#include <iostream>

namespace aphid {

namespace topo {

PathData::PathData() :
_nv(0)
{}

bool PathData::isEmpty() const
{ return _nv < 1; }

void PathData::addToSet(int iset, const Vector3F& pos)
{
	_nv++;
	
	if(m_groups.find(iset) == m_groups.end() ) {
		SetMean am;
		am._nv = 0;
		am._pos.setZero();
		m_groups[iset] = am;
	}
	
	m_groups[iset]._pos += pos;
	m_groups[iset]._nv++;
}

int PathData::numSets()
{ return m_groups.size(); }

void PathData::getSetMeans(Vector3F* dest)
{
	std::map<int, SetMean >::iterator it = m_groups.begin();
	int j = 0;
	for(;it!=m_groups.end();++it) {
		dest[j] = it->second._pos / (float)it->second._nv;
		j++;
	}
}

const float GeodesicPath::DspRootColor[3] = {1.f, 0.f, 0.f};
const float GeodesicPath::DspTipColor[8][3] = {
{0.f, 1.f, 1.f},
{1.f, 1.f, 0.f},
{.8f, 0.f, .5f},
{0.f, .5f, .8f},
{.5f, .8f, 0.f},
{.4f, .2f, 1.f},
{.2f, 1.f, .4f},
{1.f, .2f, .4f},
};

GeodesicPath::GeodesicPath() :
m_numVertices(0),
m_numJoints(0)
{}

GeodesicPath::~GeodesicPath()
{}

void GeodesicPath::create(int nv)
{
	m_dist2Root.reset(new float[nv]);
	m_dysCols.reset(new float[nv * 3]);
	m_pathInd.reset(new int[nv]);
	memset(m_pathInd.get(), 0, nv<<2);
	m_distDiff.reset(new float[nv]);
	const float defCol[3] = {0.f, .35f, .45f};
	for(int i=0;i<nv;++i) {
		memcpy(&m_dysCols[i*3], defCol, 12 );
		m_distDiff[i] = 1.f;
	}
	m_numVertices = nv;
}

const std::deque<int>& GeodesicPath::rootNodeIndices() const
{ return m_rootNodeIndices; }

const std::deque<int>& GeodesicPath::tipNodeIndices() const
{ return m_tipIndices; }	

float* GeodesicPath::distanceToRoot()
{ return m_dist2Root.get(); }

float* GeodesicPath::distanceToLastTip()
{ return m_dist2Tip.back()->get(); }

int GeodesicPath::lastTipNodeIndex()
{ return m_tipIndices.back(); }

int GeodesicPath::numRoots() const
{ return m_rootNodeIndices.size(); }

int GeodesicPath::numTips() const
{ return m_tipIndices.size(); }

bool GeodesicPath::hasRoot() const
{ return m_rootNodeIndices.size() > 0; }

bool GeodesicPath::hasTip() const
{ return m_tipIndices.size() > 0; }

const int& GeodesicPath::numVertices() const
{ return m_numVertices; }

const int& GeodesicPath::numJoints() const
{ return m_numJoints; }

const Vector3F* GeodesicPath::jointPos() const
{ return m_jointPos.get(); }

const float* GeodesicPath::dysCols() const
{ return m_dysCols.get(); }

const float* GeodesicPath::dspRootColR() const
{ return DspRootColor; }

const float* GeodesicPath::dspTipColR(int i) const
{ return DspTipColor[i&7]; }

void GeodesicPath::addRoot(int x)
{ m_rootNodeIndices.push_back(x); }

void GeodesicPath::addTip(int x)
{
	m_tipIndices.push_back(x);
	FltArrTyp* ad = new FltArrTyp;
	ad->reset(new float[m_numVertices] );
	m_dist2Tip.push_back(ad);
}

void GeodesicPath::setLastRootNodeIndex(int x)
{ m_rootNodeIndices.back() = x; }

void GeodesicPath::setLastTipNodeIndex(int x)
{ m_tipIndices.back() = x; }

void GeodesicPath::clearAllPath()
{
    if(m_tipIndices.size() < 1)
        return;
		
    m_tipIndices.clear();
    std::deque<FltArrTyp* >::iterator it = m_dist2Tip.begin();
    for(;it!=m_dist2Tip.end();++it) {
        delete *it;
    }
    m_dist2Tip.clear();
	for(int i=0;i<m_numVertices;++i) {
		m_distDiff[i] = 1.f;
	}
	m_numJoints = 0;
}

void GeodesicPath::findPathToTip()
{
    if(m_tipIndices.size() < 1)
        return;
    
    const int itip = m_tipIndices.size() - 1;
    const float* tipCol = DspTipColor[itip & 7];
    const int& tipNode = m_tipIndices.back();
    const float* distT = m_dist2Tip.back()->get();
    const float* distR = m_dist2Root.get();
    const float& pthl = distR[tipNode];
    for(int i=0;i<m_numVertices;++i) {
        float diff = (distT[i] + distR[i] - pthl) / pthl;
		if(diff < 0.f)
			diff = -diff;
			
		if(diff < .1f || m_distDiff[i] > .99f) {
		//if(diff < .1f && diff < m_distDiff[i]) {
		    m_pathInd[i] = itip;
			m_distDiff[i] = diff;
			float* ci = &m_dysCols[i*3];
		    memcpy(ci, tipCol, 12);
		}
	}
}

void GeodesicPath::colorByDistanceToRoot(const float& maxD)
{
	for(int i=0;i<m_numVertices;++i) {
		float* ci = &m_dysCols[i*3];
		if(m_dist2Root[i] > maxD) {
			memset(ci, 0, 12 );
		} else {
			ci[1] = m_dist2Root[i] / maxD;
			ci[0] = 1.f - ci[1];
			ci[2] = 0.f;
		}
	}
}

bool GeodesicPath::build(const float& unitD,
					const Vector3F* pos)
{
	const int np = numTips();
	std::cout<<"\n build "<<np<<" path";
	
	PathData* pds = new PathData[np];
	
	const float* distR = m_dist2Root.get();
    const int* pathI = m_pathInd.get();
	
	for(int i=0;i<m_numVertices;++i) {
		const int& phi = pathI[i];
		if(phi >= np) {
			std::cerr<<"\n ERROR oor path ind v"<<i;
			continue;
		}
		
		const int setI = distR[i] / unitD;
		pds[phi].addToSet(setI, pos[i]);
		
	}
	
	bool stat = true;
	for(int i=0;i<np;++i) {
		if(pds[i].isEmpty() ) {
			std::cerr<<"\n ERROR path["<<i<<"] is empty";
			stat = false; 
		}
	}
	
	if(stat)
		stat = buildSkeleton(pds);
	
	delete[] pds;
	return stat;
}

bool GeodesicPath::buildSkeleton(PathData* pds)
{
	const int np = numTips();
	m_jointCounts.reset(new int[np]);
	m_jointBegins.reset(new int[np + 1]);
	
	m_numJoints = 0;
	for(int i=0;i<np;++i) {
		const int ns = pds[i].numSets();
		m_jointCounts[i] = ns;
		m_jointBegins[i] = m_numJoints;
		m_numJoints += ns;
	}
	m_jointBegins[np] = m_numJoints;
	
	m_jointPos.reset(new Vector3F[m_numJoints]);
	for(int i=0;i<np;++i) {
		pds[i].getSetMeans(&m_jointPos[m_jointBegins[i]]);
	}
	return true;
}

}

}
