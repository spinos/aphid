/*
 *  H5Holder.h
 *  opium
 *
 *  Created by jian zhang on 1/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <H5Availability.h>
#include <SampleFrame.h>

namespace aphid {
    
class H5Holder {

	SampleFrame m_sampler;
	double m_lastTime;
	bool m_hasSampler;
	
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
	
private:

};

}