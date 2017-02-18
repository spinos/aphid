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

#include <math/BoundingBox.h>
#include <SelectionContext.h>

namespace aphid {

class SampleFilter {

	BoundingBox m_bbox;
	Vector3F m_center;
	float m_radius;
	SelectionContext::SelectMode m_mode;
	
public:
	SampleFilter();
	virtual ~SampleFilter();
	
	void setMode(SelectionContext::SelectMode mode);
	void setSphere(const Vector3F & center,
						const float & radius);
	void limitBox(const BoundingBox & b);
	
	bool intersect(const BoundingBox & b) const;
	bool intersect(const Vector3F & p) const;
	
	Vector3F boxLow() const;
	Vector3F boxHigh() const;
	
	bool isReplacing() const;
	bool isRemoving() const;
	bool isAppending() const;
	
protected:

private:
};

}
#endif