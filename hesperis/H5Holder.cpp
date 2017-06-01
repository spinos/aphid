/*
 *  H5Holder.cpp
 *  opium
 *
 *  Created by jian zhang on 1/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "H5Holder.h"
#include <h5/HBase.h>
#include <h5/HFrameRange.h>

namespace aphid {
    
H5Availability H5Holder::H5Files;

H5Holder::H5Holder() : m_lastFilename(""),
m_lastTime(1e28), m_hasSampler(false), m_hasSpfSegment(false) 
{}

H5Holder::~H5Holder() 
{}

void H5Holder::readSampler(SampleFrame & sampler)
{
	bool hasSpf = false;
    HBase w("/");
    if(w.hasNamedAttr(".spf")) {
        w.readIntAttr(".spf", & sampler.m_spf);
        hasSpf = true;
    }
    w.close();
	
    if(hasSpf) {
		std::cout<<"\n spf "<<sampler.m_spf;
		return;
	}
	
	if(!HObject::FileIO.checkExist("/.spf" )) {
	    return;
	}
    
    HIntAttribute aspf("/.spf");
    if(aspf.open()) {
        aspf.read(& sampler.m_spf);
        aspf.close();
    }
}

bool H5Holder::openH5File(const std::string & fileName)
{ 
	if(!H5Files.openFile(fileName.c_str(), HDocument::oReadOnly ) ) {
	    return false; 
	}
/// to read sampler on name changed
	if(m_lastFilename != fileName) {
	    m_hasSampler = false;
	    m_lastFilename = fileName;
	}
	
	if(!m_hasSampler) {
		readSampler(m_sampler);
		m_hasSampler = true;
	}
    
    m_hasSpfSegment = readSpfSegment();
    
	return true;
}

void H5Holder::setTime(double x)
{ m_lastTime = x; }

bool H5Holder::isTimeChanged(double x) const
{ return x != m_lastTime; }

int H5Holder::getFrame(double x, int tmin, int tmax) const
{
	int r = float(int(x * 1000.f + 0.5f))/1000.f;
	if(r < tmin) r = tmin;
	if(r > tmax) r = tmax;
	return r;
}

SampleFrame * H5Holder::sampler()
{ return &m_sampler; }

bool H5Holder::readSpfSegment()
{
    int oldFirst = AFrameRange::FirstFrame;
    int oldLast = AFrameRange::LastFrame;
    int oldSpf = AFrameRange::SamplesPerFrame;
    float oldFps = AFrameRange::FramesPerSecond;
    
    AFrameRange afr;
    HFrameRange fr(".fr");
    fr.load(&afr);
    fr.close();
    
    AFrameRange::FirstFrame = oldFirst;
    AFrameRange::LastFrame = oldLast;
    AFrameRange::SamplesPerFrame = oldSpf;
    AFrameRange::FramesPerSecond = oldFps;
    
    if(afr.segmentExpr().size() < 3) {
        return false;
     }
    
    if( m_spfSegment.create(afr.segmentExpr() ) ) {
        return true;
     }
    
    return false;
}

const AFrameRangeSegment & H5Holder::spfSegment() const
{ return m_spfSegment; }

const bool & H5Holder::hasSpfSegment() const
{ return m_hasSpfSegment; }

}
