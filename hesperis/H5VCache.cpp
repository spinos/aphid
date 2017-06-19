/*
 *  H5VCache.cpp
 *  opium
 *
 *  Created by jian zhang on 3/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "H5VCache.h"
#include <math/BaseBuffer.h>
#include <h5/HBase.h>
#include <h5/VerticesHDataset.h>
#include <sstream>
#include <foundation/SHelper.h>
#include <boost/lexical_cast.hpp>

namespace aphid {

H5VCache::H5VCache() :
m_hasPiecesChecked(false),
m_hasArbitrarySampleChecked(false),
m_hasArbitrarySamples(false)
{
	m_data[0] = new BaseBuffer;
    m_data[1] = new BaseBuffer;
    m_counts = new BaseBuffer;
    m_drifts = new BaseBuffer;
    m_numPieces = 1;
    m_numPoints = 0;
    m_minInd = 1<<30;
	m_blender = 1e8f;
}

H5VCache::~H5VCache()
{
	delete m_data[0];
    delete m_data[1];
    delete m_counts;
    delete m_drifts;
}

void H5VCache::checkPieces(const std::string & pathName, unsigned numIterPoints)
{
    HBase geoGrp(pathName);
    if(geoGrp.hasNamedData(".cc") && geoGrp.hasNamedAttr(".nc")) {
        int nc; 
        geoGrp.readIntAttr(".nc", &nc);
        std::cout<<"\n H5VCache read n curves "<< nc;
        
        if(m_counts->bufferSize() != nc * 4) {
            std::cout<<"\n init curve drift "<<pathName.c_str();
            m_counts->create(nc * 4);
            m_drifts->create(nc * 4);
            geoGrp.readIntData(".cc", nc, (int *)m_counts->data());
            scanExclusive(nc, (int *)m_drifts->data(), (int *)m_counts->data());
        }
        m_numPieces = nc;
        geoGrp.readIntAttr(".nv", &m_numPoints);
    }
    else if(geoGrp.hasNamedData(".pntdrift") && geoGrp.hasNamedAttr(".npart")) {
        int nm;
        geoGrp.readIntAttr(".npart", &nm);
        std::cout<<"\n H5VCache read n meshes "<<nm;
        
        if(m_drifts->bufferSize() != nm * 4) {
            std::cout<<"\n init mesh drift "<<pathName.c_str();
            m_counts->create(nm * 4);
            m_drifts->create(nm * 4);
            geoGrp.readIntData(".pntdrift", nm, (int *)m_drifts->data());
        }
        m_numPieces = nm;
        geoGrp.readIntAttr(".nv", &m_numPoints);
    }
    else {
		m_numPoints = numIterPoints;
		m_numPieces = 1; 
    }
    geoGrp.close();
    m_hasPiecesChecked = true;
	
	m_data[0]->create(m_numPoints * 12);
    m_data[1]->create(m_numPoints * 12);
}

void H5VCache::scanExclusive(int n, int * dst, int * src)
{
    dst[0] = 0;
    for(int i = 1; i < n; i++) {
        dst[i] = src[i-1] + dst[i-1];
	}
}

int H5VCache::driftIndex(int idx) const
{
    if((idx << 2) >= m_drifts->bufferSize() ) {
		return 0;
	}
	
    return ((int *)m_drifts->data())[idx];
}

char H5VCache::readFrame(float *data, int count, const char *h5name, int frame, int sample,
						bool isLegacy)
{	
	if(isLegacy) {
		return readFrameLegacy(data, count, h5name, frame, sample);
	}
	
    char res = 0;
    
    std::stringstream sst;
	sst.str("");
	sst<<h5name<<"/.geom/.bake/";

    HBase bakeNode(sst.str().c_str());
    
    sst.str("");
    sst<<frame;
    if(sample > 0) {
		sst<<"."<<sample;
	}
    
    if(bakeNode.hasNamedData(sst.str().c_str())) {
        res = bakeNode.readVector3Data(sst.str().c_str(), count, (Vector3F *)data);
    }
    else {
        std::cout<<"\n H5VCache cannot open vertices data space of bake "<< h5name<< " at "<< sst.str();
    }
    
    bakeNode.close();
    return res;
}

char H5VCache::readFrameLegacy(float *data, int count, const char *h5name, int frame, int sample)
{
	std::stringstream sst;
	sst.str("");
	sst<<h5name<<"/.geom/.bake/"<<frame;
	if(sample > 0) {
		sst<<"."<<sample;
	}
	
	VerticesHDataset dsetBake(sst.str());
	dsetBake.setNumVertices(count);
		
	if(!dsetBake.open()) {
		std::cout<<"\n H5VCache cannot open data space of bake "<< h5name;
		return 0;
	}
	
	char res = 0;
	
	if(dsetBake.hasEnoughSpace()) {
		dsetBake.read((char *)data);
		res = 1;
	}
	else {
		std::cout<<"\n H5VCache data space does not match "<< h5name;
	}
		
	dsetBake.close();
	
	return res;
}

void H5VCache::mixFrames(float weight0, float weight1)
{
	float * p = (float *)m_data[0]->data();
	float * p1 = (float *)m_data[1]->data();
	int n = m_numPoints * 3;
	for ( int i=0; i < n; i++) {
		p[i] = p[i] * weight0 + p1[i] * weight1;
	}
}

void H5VCache::resetMinInd()
{ m_minInd = 1<<30; }

const bool & H5VCache::hasData() const
{ return m_hasData; }

const bool & H5VCache::hasPiecesChecked() const
{ return m_hasPiecesChecked; }

const bool & H5VCache::hasArbitrarySampleChecked() const
{ return m_hasArbitrarySampleChecked; }

bool H5VCache::readData(const std::string & fileName, 
                   const std::string & pathName,
                   double dtime,
                   unsigned numIterPoints,
                   int isSubframe,
				   bool toSkipReading,
				   bool isLegacy)
{
	m_hasData = false;
    if(!openH5File(fileName ) ) {
		std::cout<<"\n H5VCache cannot open h5 file "<< fileName;
        return m_hasData;
	}
   
    if(!HObject::FileIO.checkExist(HObject::ValidPathName(pathName))) {
        std::cout<<"\n H5VCache cannot find grp "<< pathName;
        return m_hasData;
    }
    
    // std::cout<<"\n oflBakePNode switch to h5 file "<<fileName;
    
    if(!hasArbitrarySampleChecked() ) {
        checkArbitrarySamples(HObject::ValidPathName(pathName) );
    }
    
    if(!hasPiecesChecked() ) {
        checkPieces(pathName, numIterPoints);
	}
	
	if(toSkipReading) {
		m_hasData = true;
		return true;
	}
	
    int currentSpf = sampler()->m_spf;
    
    if(hasSpfSegment() ) {
        spfSegment().getSamples<double>(currentSpf, dtime);
        //aphid::AHelper::Info<std::string>(" overSpf", spfSegment().str() );
    }

    sampler()->calculateWeights(dtime, currentSpf);
    if(isSubframe == 0) {
		sampler()->m_weights[0] = 1.f;
	}
	
	findArbitrarySample(dtime);
	
	//std::cout<<sampler()->str();
	//std::cout.flush();
    
    if(!readFrame((float *)m_data[0]->data(), m_numPoints, pathName.c_str(), 
        sampler()->m_frames[0], sampler()->m_samples[0], 
			isLegacy)) {
            return m_hasData;
	}
	
	m_hasData = true;
            
    if(sampler()->m_weights[0] > 0.99f) {
		return m_hasData;
	}

    if(!readFrame((float *)m_data[1]->data(), m_numPoints, pathName.c_str(), 
        sampler()->m_frames[1], sampler()->m_samples[1],
		isLegacy)) {
            return m_hasData;
	}
            
    mixFrames(sampler()->m_weights[0], sampler()->m_weights[1]);
		
	return m_hasData;
}

float * H5VCache::data0() const
{ return (float *)m_data[0]->data(); }

const int & H5VCache::numPieces() const
{ return m_numPieces; }

const int & H5VCache::numPoints() const
{ return m_numPoints; }

void H5VCache::setMinInd(int x)
{ m_minInd = x; }

bool H5VCache::isFirstPiece(int x) const
{ return x < m_minInd; }

void H5VCache::mixData(const H5VCache * b, float alpha)
{	
	if(alpha < 1e-2f) {
		return;
	}
	
	if(!b->hasData() ) {
		std::cout<<"\n H5VCache::mixData has no data ";
		return;
	}
	
	if(b->numPoints() < numPoints() ) {
		std::cout<<"\n H5VCache::mixData not enough data "<<b->numPoints()
				<<"\n need "<<numPoints();
		return;
	}
	
	float * p = (float *)m_data[0]->data();
	float * p1 = b->data0();
	int n = m_numPoints * 3;
	
	if(alpha > .99f) {
		memcpy(p, p1, n<<2);
		return;
	}
	
	const float beta = 1.f - alpha;
	for ( int i=0; i < n; i++) {
		p[i] = p[i] * beta + p1[i] * alpha;
	}
	
}

bool H5VCache::isBlenderChanged(float x) const
{ return x != m_blender; }
	
void H5VCache::setBlender(float x)
{ m_blender = x; }

void H5VCache::checkArbitrarySamples(const std::string & pathName)
{
    m_arbitrarySamples.clear();
    m_hasArbitrarySamples = false;
    std::cout<<"\n check arbitrary samples in "<<pathName;
    std::stringstream sst;
	sst.str("");
	sst<<pathName<<"/.geom/.bake/";

    HBase bakeNode(sst.str().c_str());
    const int nc = bakeNode.numChildren();
		
		for(int i = 0;i<nc;i++) {
			if(bakeNode.isChildData(i)) {
			    addASample(bakeNode.getChildName(i) );
				
			}
		}
		
		bakeNode.close();
    
    const int ns = m_arbitrarySamples.sampleCount();
    if(ns > 0) {
        std::cout<<"\n n arbitrary sample "<<ns;
        m_hasArbitrarySamples = true;
     }
    std::cout.flush();
    
    m_hasArbitrarySampleChecked = true;
}

bool H5VCache::asSampleTime(int& frame, int& subframe, 
                const std::string& stime) const
{
    std::string atime(stime);
    if(!SHelper::PartAfterDot(atime) ) {
        return false;
    }
    
    try {
    subframe = boost::lexical_cast<int>(atime);
    
    float ftime = boost::lexical_cast<float>(stime);
    frame = ftime;
    if(ftime < 0.f) {
        frame--;
     }
    } catch (...) {
        std::cout<<" H5VCache::asSampleTime cannot cast "<<atime;
        return false;
    }
    
    return true;
}

void H5VCache::addASample(const std::string & stime)
{
    int iframe, subframe;
    if(!asSampleTime(iframe, subframe, stime ) ) {
        return;
    }
    if(subframe < 99) {
        return;
    }
// std::cout<<"\n frame subframe "<<iframe<<" "<<subframe<<" "<<stime;
    m_arbitrarySamples.insertSample(iframe, subframe);
}

const bool& H5VCache::hasArbitrarySamples() const
{ return m_hasArbitrarySamples; }

bool H5VCache::findArbitrarySample(const double& dtime)
{
    if(!m_hasArbitrarySamples ) {
        return false;
    }
    int iframe = double(int(dtime * 1000 + 0.5) )/1000.0;
    if(dtime < 0.0) {
        iframe--;
    }
    
    std::cout<<"\n search subframe "<<dtime<<" "<<iframe;
    
    SubframeSampleTimeI * sps = m_arbitrarySamples.findFrame(iframe);
    if(!sps) {
        return false;
    }
    
    int subframe = 1000000 * (dtime - iframe);
    if(subframe < 99) {
        return false;
    }
    
    std::cout<<"\n search subframe "<<subframe;
    
    int curk, prek = 9999999;
    sps->begin();
    while(!sps->end() ) {
        curk = sps->key();
        
        if(subframe <= curk && prek > 1000000) {
            prek = curk;
            break;
        }
        if(subframe > prek && subframe <= curk ) {
            break;
        }
        prek = curk;
        sps->next();
    }

/// first or last one
    if(prek == curk) {
        subframe = prek;
    } else {

/// closer one
    int diff0 = subframe - prek;
    int diff1 = curk - subframe;
    if(diff0 < diff1) {
        subframe = prek;
    } else {
        subframe = curk;
    }
    
    }
    
    sampler()->setFirst(iframe, subframe, 1.f);

    return true;
}

}
