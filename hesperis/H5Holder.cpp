/*
 *  H5Holder.cpp
 *  opium
 *
 *  Created by jian zhang on 1/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "H5Holder.h"
#include <HBase.h>

H5Availability H5Holder::H5Files;

H5Holder::H5Holder() : m_lastTime(1e28), m_hasSampler(false) {}
H5Holder::~H5Holder() {}

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
	
	if(!HObject::FileIO.checkExist("/.spf" )) return;
    
    HIntAttribute aspf("/.spf");
    if(aspf.open()) {
        aspf.read(& sampler.m_spf);
        aspf.close();
    }
}

bool H5Holder::openH5File(const std::string & fileName)
{ 
	if(!H5Files.openFile(fileName.c_str(), HDocument::oReadAndWrite ) ) return false; 
	if(!m_hasSampler) {
		readSampler(m_sampler);
		m_hasSampler = true;
	}
	return true;
}

void H5Holder::setTime(double x)
{ m_lastTime = x; }

bool H5Holder::isTimeChanged(double x) const
{ return x != m_lastTime; }

int H5Holder::getFrame(double x, int tmin, int tmax) const
{
	int r = float(int(x * 1000 + 0.5))/1000.f;
	if(r < tmin) r = tmin;
	if(r > tmax) r = tmax;
	return r;
}

SampleFrame * H5Holder::sampler()
{ return &m_sampler; }
