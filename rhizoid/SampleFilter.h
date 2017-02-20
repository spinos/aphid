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
#include <math/ANoise3.h>

namespace aphid {
    
class ExrImage;

class SampleFilter : public ANoise3Sampler {

	BoundingBox m_bbox;
	Vector3F m_center;
	float m_radius;
	int m_maxSampleLevel;
	float m_sampleGridSize;
	float m_portion;
	SelectionContext::SelectMode m_mode;
	
public:
	SampleFilter();
	virtual ~SampleFilter();
	
	void setMode(SelectionContext::SelectMode mode);
	void setPortion(const float & x);
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
	
	const int & maxSampleLevel() const;
	const float & sampleGridSize() const;
	void computeGridLevelSize(const float & cellSize,
				const float & sampleDistance);
				
	bool throughPortion(const float & x) const;
	bool throughNoise3D(const Vector3F & p) const;
	bool throughImage(const float & k, const float & s, const float & t) const;
	
	const float & portion() const;
	
	const ExrImage * m_imageSampler;
	
protected:

private:
};

}
#endif