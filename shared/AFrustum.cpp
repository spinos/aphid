/*
 *  AFrustum.cpp
 *  
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "AFrustum.h"
#include <iostream>
void AFrustum::set(const float & hfov,
			const float & aspectRatio,
			const float & clipNear,
			const float & clipFar,
			const Matrix44F & mat)
{
	float nearRight = clipNear * hfov;
	float nearLeft = -nearRight;
	float nearUp = nearRight * aspectRatio;
	float nearBottom = -nearUp;
	float farRight = clipFar * hfov;
	float farLeft = -farRight;
	float farUp = farRight * aspectRatio;
	float farBottom = -farUp;
	
	m_v[0] = mat.transform(Vector3F(nearLeft, nearBottom, clipNear));
	m_v[1] = mat.transform(Vector3F(nearRight, nearBottom, clipNear));
	m_v[2] = mat.transform(Vector3F(nearRight, nearUp, clipNear));
	m_v[3] = mat.transform(Vector3F(nearLeft, nearUp, clipNear));
	m_v[4] = mat.transform(Vector3F(farLeft, farBottom, clipFar));
	m_v[5] = mat.transform(Vector3F(farRight, farBottom, clipFar));
	m_v[6] = mat.transform(Vector3F(farRight, farUp, clipFar));
	m_v[7] = mat.transform(Vector3F(farLeft, farUp, clipFar));
}

void AFrustum::setOrtho(const float & hwith,
			const float & aspectRatio,
			const float & clipNear,
			const float & clipFar,
			const Matrix44F & mat)
{
	float hUp = hwith * aspectRatio;
	
	m_v[0] = mat.transform(Vector3F(-hwith, -hUp, clipNear));
	m_v[1] = mat.transform(Vector3F(hwith, -hUp, clipNear));
	m_v[2] = mat.transform(Vector3F(hwith, hUp, clipNear));
	m_v[3] = mat.transform(Vector3F(-hwith, hUp, clipNear));
	m_v[4] = mat.transform(Vector3F(-hwith, -hUp, clipFar));
	m_v[5] = mat.transform(Vector3F(hwith, -hUp, clipFar));
	m_v[6] = mat.transform(Vector3F(hwith, hUp, clipFar));
	m_v[7] = mat.transform(Vector3F(-hwith, hUp, clipFar));
}

const Vector3F * AFrustum::v(int idx) const
{ return &m_v[idx]; }

Vector3F AFrustum::X(int idx) const
{ m_v[idx]; }

Vector3F AFrustum::supportPoint(const Vector3F & v, Vector3F * localP) const
{
	float maxdotv = -1e8f;
    float dotv;
	
    Vector3F res, q;
    for(int i=0; i < 8; i++) {
        q = m_v[i];
        dotv = q.dot(v);
        if(dotv > maxdotv) {
            maxdotv = dotv;
            res = q;
            if(localP) *localP = q;
        }
    }
    
    return res;
}
