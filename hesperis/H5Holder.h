/*
 *  H5Holder.h
 *  opium
 *
 *  Created by jian zhang on 1/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <h5/H5Availability.h>
#include <SampleFrame.h>
#include <AFrameRange.h>

namespace aphid {
    
class H5Holder {

    AFrameRangeSegment m_spfSegment;
	SampleFrame m_sampler;
	double m_lastTime;
	bool m_hasSampler;
    bool m_hasSpfSegment;
	
public:
	H5Holder();
	virtual ~H5Holder();
	
	static H5Availability H5Files;
	
protected:
	void readSampler(SampleFrame & sampler);
	bool openH5File(const std::string & fileName);
	void setTime(double x);
	bool isTimeChanged(double x) const;
	int getFrame(double x, int tmin, int tmax) const;
	SampleFrame * sampler();
	bool readSpfSegment();
    
    const AFrameRangeSegment & spfSegment() const;
    const bool & hasSpfSegment() const;
    
private:

};

}