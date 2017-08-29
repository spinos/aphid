/*
 *  SampleFilter.h
 *  
 *  through portion and facing angle 
 *
 *  Created by jian zhang on 7/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SMP_SAMPLE_FILTER_H
#define APH_SMP_SAMPLE_FILTER_H

#include <math/Vector3F.h>
#include <boost/scoped_array.hpp>

namespace aphid {

namespace smp {

class SampleFilter {

	boost::scoped_array<Vector3F > m_samples;
	int m_numFilteredSamples;
	Vector3F m_facing;
	float m_portion;
	float m_angle;
	int m_maxNumSample;
	
public:
	SampleFilter();
	virtual ~SampleFilter();
	
	const int& numFilteredSamples() const;
	const Vector3F* filteredSamples() const;
	
	void setPortion(const float& x);
	void setFacing(const Vector3F& v);
	void setAngle(const float& x);
	void setNumSampleLimit(const int& x);
	
	template<typename T>
	void processFilter(T* grd);

/// by distance to origin	
	void sortSamples();
	
protected:

private:
	bool isFiltered(const Vector3F& v) const;
	
};

template<typename T>
void SampleFilter::processFilter(T* grd)
{
	const Vector3F * poss = grd->positions(); 
	const int np = grd->numSamples();
	
	m_samples.reset(new Vector3F[np]);
	int acc = 0;
	for(int i=0;i<np;++i) {
		if(isFiltered(poss[i]) ) 
			continue;
	    m_samples[acc++] = poss[i];
	}
	m_numFilteredSamples = acc;
	
	sortSamples();
	
	if(m_numFilteredSamples > m_maxNumSample)
		m_numFilteredSamples = m_maxNumSample;
}

}

}

#endif