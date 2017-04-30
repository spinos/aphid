/*
 *  LODFn.h
 *  aphid
 *
 *  Created by jian zhang on 1/3/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>

class LODFn {
public:
	LODFn();
	void setEyePosition(const Vector3F & pos);
	void setFieldOfView(const float & fov);
	float computeLOD(const Vector3F & p, const float r, const unsigned npix) const;
	void setOverall(float x);
	float overall() const;
protected:

private:
	Vector3F m_eye;
	float m_fov, m_overall;
};