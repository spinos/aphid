/*
 *  SampleFilter.h
 *  
 *
 *  Created by jian zhang on 2/18/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SAMPLE_FILTER_H
#define APH_SAMPLE_FILTER_H

#include <math/Vector3F.h>
#include <SelectionContext.h>

namespace aphid {

class SampleFilter {

	Vector3F m_center;
	float m_radius;
	SelectionContext::SelectMode m_mode;
	
public:
	SampleFilter();
	virtual ~SampleFilter();
	
	void setMode(SelectionContext::SelectMode mode);
	void setSphere(const Vector3F & center,
						const float & radius);
	
	bool insideSphere(const Vector3F & p) const;
	Vector3F boxLow() const;
	Vector3F boxHigh() const;
	
	bool isRemoving() const;
	
protected:

private:
};

}
#endif