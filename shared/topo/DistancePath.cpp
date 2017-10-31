/*
 *  DistancePath.cpp
 * 
 *  Created by jian zhang on 10/26/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DistancePath.h"
#include <iostream>

namespace aphid {

namespace topo {

PathData::PathData() :
_nv(0)
{}

void PathData::addToSet(int iset, const Vector3F& pos,
					const float& val)
{
	_nv++;
	
	if(m_groups.find(iset) == m_groups.end() ) {
		SetMean am;
		am._nv = 0;
		am._pos.setZero();
		am._val = 0.f;
		m_groups[iset] = am;
	}
	
	SetMean& sm = m_groups[iset];
	sm._pos += pos;
	sm._val += val;
	sm._nv++;
}

int PathData::numSets()
{ return m_groups.size(); }

void PathData::average()
{
	std::map<int, SetMean >::iterator it = m_groups.begin();
	int j = 0;
	for(;it!=m_groups.end();++it) {
		it->second._pos /= (float)it->second._nv;
		it->second._val /= (float)it->second._nv;
		j++;
	}
}

void PathData::getSetPos(Vector3F* dest)
{
	std::map<int, SetMean >::iterator it = m_groups.begin();
	int j = 0;
	for(;it!=m_groups.end();++it) {
		dest[j] = it->second._pos;
		j++;
	}
}

int PathData::getClosestSet(const float& val)
{
	std::map<int, SetMean >::iterator it = m_groups.begin();
	int r = 0, j = 0;
	float minD = 1e8f;
	for(;it!=m_groups.end();++it) {
		float d = it->second._val - val;
		if(d < 0.f)
			d = -d;
		
		if(minD > d) {
			minD = d;
			r = j;
		}
		j++;
	}
	return r;
}

const float DistancePath::DspRegionColor[8][3] = {
{1.f, 0.f, 0.f},
{0.f, 1.f, 0.f},
{0.f, 0.f, 1.f},
{1.f, 1.f, 0.f},
{0.f, 1.f, 1.f},
{1.f, 0.f, 1.f},
{.8f, .5f, 0.f},
{0.f, .5f, .8f},
};

DistancePath::DistancePath() :
m_numVertices(0)
{}

DistancePath::~DistancePath()
{}

void DistancePath::create(int nv)
{
	m_dysCols.reset(new float[nv * 3]);
	m_pathLab.reset(new int[nv]);
	m_setInd.reset(new int[nv]);
	memset(m_pathLab.get(), 0, nv<<2);
	m_distDiff.reset(new float[nv]);
	const float defCol[3] = {0.f, .35f, .45f};
	for(int i=0;i<nv;++i) {
		memcpy(&m_dysCols[i*3], defCol, 12 );
		m_distDiff[i] = 1e8f;
	}
	m_numVertices = nv;
	addSite(0);
}

int DistancePath::numRegions() const
{ return m_siteIndices.size(); }

int DistancePath::numSeeds() const
{ return numRegions() - 1; }

const int& DistancePath::siteNodeIndex(int i) const
{ return m_siteIndices[i]; }

const int& DistancePath::rootNodeInd() const
{ return m_siteIndices[0]; }

const int& DistancePath::seedNodeIndex(int i) const
{ return m_siteIndices[i+1]; }

float* DistancePath::distanceToRoot()
{ return m_dist2Site[0]->get(); }

float* DistancePath::distanceToSite(int i)
{ return m_dist2Site[i]->get(); }

float* DistancePath::distanceToSeed(int i)
{ return m_dist2Site[i+1]->get(); }

int DistancePath::lastTipNodeIndex()
{ return m_siteIndices.back(); }

const int& DistancePath::numVertices() const
{ return m_numVertices; }

const float* DistancePath::dysCols() const
{ return m_dysCols.get(); }

float* DistancePath::vertexCols()
{ return m_dysCols.get(); }

const float* DistancePath::dspRegionColR(int i) const
{ return DspRegionColor[i&7];}

void DistancePath::setRootNodeIndex(int x)
{ m_siteIndices[0] = x; }

void DistancePath::addSite(int x)
{
	FltArrTyp* ad = new FltArrTyp;
	ad->reset(new float[m_numVertices] );
	m_dist2Site.push_back(ad);
	m_siteIndices.push_back(x);
}

void DistancePath::addSeed(int x)
{
	addSite(x);
	//m_pathLength.push_back(distanceToRoot()[x]);
}

void DistancePath::setLastTipNodeIndex(int x)
{ m_siteIndices.back() = x; }

void DistancePath::clearAllPath()
{
    if(m_siteIndices.size() < 2)
        return;
		
    m_siteIndices.clear();
    std::deque<FltArrTyp* >::iterator it = m_dist2Site.begin();
    for(;it!=m_dist2Site.end();++it) {
        delete *it;
    }
    m_dist2Site.clear();
	for(int i=0;i<m_numVertices;++i) {
		m_distDiff[i] = 1e8f;
	}
	m_pathLength.clear();
	addSite(0);
}

void DistancePath::colorByRegionLabels()
{
	const int maxLab = numSeeds();
	for(int i=0;i<m_numVertices;++i) {
		float* ci = &m_dysCols[i*3];
		const int& labi = vertexLabels()[i];
		if(labi < 0 || labi > maxLab) {
			memset(ci, 0, 12 );
		} else {
			memcpy(ci, dspRegionColR(labi), 12 );
		}
	}
}

void DistancePath::colorByDistanceToRoot(const float& maxD)
{
	for(int i=0;i<m_numVertices;++i) {
		float* ci = &m_dysCols[i*3];
		if(distanceToRoot()[i] > maxD) {
			memset(ci, 0, 12 );
		} else {
			ci[1] = distanceToRoot()[i] / maxD;
			ci[0] = 1.f - ci[1];
			ci[2] = 0.f;
		}
	}
}

bool DistancePath::buildLevelSet(PathData* dest,
					const float& unitD,
					const Vector3F* pos)
{
	const int np = numRegions();
	
	const float* distR = m_dist2Site[0]->get();
    const int* pathI = m_pathLab.get();
	
	for(int i=0;i<m_numVertices;++i) {
		const int& phi = pathI[i];
		if(phi >= np) {
			std::cerr<<"\n ERROR oor path ind v"<<i;
			continue;
		}
		
		const int setI = distR[i] / unitD;
		dest[phi].addToSet(setI, pos[i], distR[i]);
		
	}
	
	bool stat = true;
	for(int i=0;i<np;++i) {
		if(dest[i].numSets() < 3 ) {
			std::cerr<<"\n ERROR path["<<i<<"] is incomplete";
			stat = false; 
		} else {
			dest[i].average();
		}
	}
	
	if(stat) {
		setVertexSetInd(dest);
	}
	
	return stat;
}

void DistancePath::setVertexSetInd(PathData* dest)
{
	const float* distR = m_dist2Site[0]->get();
    const int* pathI = m_pathLab.get();
	
	for(int i=0;i<m_numVertices;++i) {
		const int& phi = pathI[i];
		m_setInd[i] = dest[phi].getClosestSet(distR[i]);
	}
}

int* DistancePath::vertexLabels()
{ return m_pathLab.get(); }

const int* DistancePath::vertexPathInd() const
{ return m_pathLab.get(); }
	
const int* DistancePath::vertexSetInd() const
{ return m_setInd.get(); }

int* DistancePath::vertexSetInd()
{ return m_setInd.get(); }

void DistancePath::labelRootAndSeedPoints()
{
	for(int i=0;i<m_numVertices;++i) {
		m_pathLab[i] = -1;
	}
	
	const int ns = m_siteIndices.size();
	for(int i=0;i<ns;++i) {
		m_pathLab[m_siteIndices[i]] = i;
	}
}

void DistancePath::collectRegionVertices(std::vector<int >& dest, 
							const int& j) const
{
	for(int i=0;i<m_numVertices;++i) {
		const int& phi = m_pathLab[i];
		if(phi == j)
			dest.push_back(i);
	}
}

}

}
