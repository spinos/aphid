/*
 *  GrowthSample.h
 *  garden
 *
 *  generate growth point and direction
 *
 *  Created by jian zhang on 7/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_GROWTH_SAMPLE_H
#define GAR_GROWTH_SAMPLE_H

#include <smp/SampleFilter.h>
#include <math/Matrix44F.h>
#include <math/Ray.h>

struct GrowthSampleProfile {

	float m_angle;
	float m_portion;
	float m_sizing;
	float m_tilt;
};

class GrowthSample : public aphid::smp::SampleFilter {

/// point and direction
	boost::scoped_array<aphid::Ray > m_pnds;
	
public:
	GrowthSample();
	virtual ~GrowthSample();
	
	void sampleBush(const GrowthSampleProfile& prof);
	
	const int& numGrowthSamples() const;
	aphid::Matrix44F getGrowSpace(const int& i,
				const GrowthSampleProfile& prof) const;
	const aphid::Vector3F& growthPoint(const int& i) const;
	const aphid::Vector3F& growthDirection(const int& i) const;
	
protected:

private:

};

#endif