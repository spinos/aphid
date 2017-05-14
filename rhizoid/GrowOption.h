/*
 *  GrowOption.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_RHIZ_GROW_OPTION_H
#define APH_RHIZ_GROW_OPTION_H

#include <math/Vector3F.h>
#include <math/ANoise3.h>

namespace aphid {

class ExrImage;

class GrowOption : public ANoise3Sampler {

	ExrImage * m_sampler;

public:
	Vector3F m_upDirection;
	Vector3F m_centerPoint;
	int m_plantId;
	float m_minScale, m_maxScale;
	float m_minMarginSize, m_maxMarginSize;
	float m_rotateNoise;
	float m_strength;
	float m_radius;
	bool m_alongNormal;
	bool m_multiGrow;
	bool m_stickToGround;
	bool m_isInjectingParticle;
	float m_strokeMagnitude;
	float m_brushFalloff;
	float m_zenithNoise;
	
	GrowOption();
	~GrowOption();
	
	void setStrokeMagnitude(const float & x);
	bool openImage(const std::string & fileName);
	void closeImage();
	bool hasSampler() const;
	std::string imageName() const;
	void sampleRed(float * col, const float & u,
					const float & v) const;
	
	const ExrImage * imageSampler() const;
/// limit to [0,3]
	void setbrushFalloff(const float & x);
	Vector3F getModifiedUpDirection() const;
	
};

}
#endif
