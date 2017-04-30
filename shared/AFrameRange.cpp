/*
 *  AFrameRange.cpp
 *  
 *
 *  Created by jian zhang on 7/2/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AFrameRange.h"
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <sstream>

namespace aphid {

int AFrameRange::FirstFrame = 0;
int AFrameRange::LastFrame = 0;
int AFrameRange::SamplesPerFrame = 1;
float AFrameRange::FramesPerSecond = 25.f;

AFrameRange::AFrameRange() {}
AFrameRange::~AFrameRange() {}
void AFrameRange::reset()
{
	FirstFrame = 0;
	LastFrame = 0;
	SamplesPerFrame = 1;
	m_segmentExpr = "";
}

bool AFrameRange::isValid() const
{ return (LastFrame > FirstFrame); }

int AFrameRange::numFramesInRange() const
{ return LastFrame - FirstFrame + 1; }

const std::string & AFrameRange::segmentExpr() const
{ return m_segmentExpr; }

std::string & AFrameRange::segmentExprRef()
{ return m_segmentExpr; }

AFrameRangeSegment::AFrameRangeSegment() {}

bool AFrameRangeSegment::create(const std::string & src)
{
    m_data.clear();
    const boost::regex re1("(?<1>-*[[:digit:]]+)/(?<2>-*[[:digit:]]+)/(?<3>-*[[:digit:]]+)");
	std::string::const_iterator start, end;
    start = src.begin();
    end = src.end();
    try {
	boost::match_results<std::string::const_iterator> what;
	while( regex_search(start, end, what, re1, boost::match_default) ) {
        
	    if(what.size() == 4) {
	    
	        RangeSegment seg;
	        seg.m_begin = boost::lexical_cast<int>(what[1]);
	        seg.m_end = boost::lexical_cast<int>(what[2]);
	        seg.m_samples = boost::lexical_cast<int>(what[3]);
	    
	        m_data.push_back(seg);
	        
		}
		start = what[0].second;
	}
	} catch(...) {
	    std::cout<<"\n AFrameRangeSegment regex error";
	    return false;
	}
    return m_data.size() > 0;
}

const std::string AFrameRangeSegment::str() const
{
    std::stringstream sst;
    sst<<"\n frame range n seg "<<m_data.size();
    std::vector<RangeSegment>::const_iterator it = m_data.begin();
    for(;it!=m_data.end();++it) {
        sst<<"\n begin "<<it->m_begin 
        <<" end "<<it->m_end
        <<" samples "<<it->m_samples;
    }
    return sst.str();
}

}

