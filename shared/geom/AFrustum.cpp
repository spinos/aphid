/*
 *  AFrustum.cpp
 *  
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *	origin at lower-left corner
 *
 *  3 - 2 near  7 - 6 far
 *  |   |       |   |
 *  0 - 1       4 - 5
 *
 */

#include "AFrustum.h"
#include <math/Matrix44F.h>
#include <geom/ConvexShape.h>
#include <iostream>

namespace aphid {

/// clip is negative in camera space
void AFrustum::set(const float & hfov,
			const float & aspectRatio,
			const float & clipNear,
			const float & clipFar,
			const Matrix44F & mat)
{
	float nearRight = -clipNear * hfov;
	float nearLeft = -nearRight;
	float nearUp = nearRight * aspectRatio;
	float nearBottom = -nearUp;
	float farRight = -clipFar * hfov;
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
{ return m_v[idx]; }

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

Vector3F AFrustum::center() const
{ return (m_v[0] * .125f +
			m_v[1] * .125f +
			m_v[2] * .125f +
			m_v[3] * .125f +
			m_v[4] * .125f +
			m_v[5] * .125f +
			m_v[6] * .125f +
			m_v[7] * .125f); }

// origin at left-up corner, center position, right and down deviation of first pixel, 
// at near and far clip
// 
//   - 1 near   - 4 far
// | 0        | 3
// 2          5
//
void AFrustum::toRayFrame(Vector3F * dst, const int & gridX, const int & gridY) const
{
    dst[1] = (m_v[2] - m_v[3]) / gridX;
    dst[2] = (m_v[0] - m_v[3]) / gridY;
	dst[0] = m_v[3] + dst[1] * .5f + dst[2] * .5f;
    
    dst[4] = (m_v[6] - m_v[7]) / gridX;
    dst[5] = (m_v[4] - m_v[7]) / gridY;
	dst[3] = m_v[7] + dst[4] * .5f + dst[5] * .5f;
	/*std::cout<<"\n grd "<<gridX<<"x"<<gridY
		<<"\n corner[0] "<<dst[0]
		<<"\n corner[1] "<<dst[0] + dst[1] * gridX
		<<"\n corner[2] "<<dst[0] + dst[2] * gridY
		<<"\n corner[3] "<<dst[0] + dst[1] * gridX + dst[2] * gridY
		<<"\n corner[4] "<<dst[3]
		<<"\n corner[5] "<<dst[3] + dst[4] * gridX
		<<"\n corner[6] "<<dst[3] + dst[5] * gridY
		<<"\n corner[7] "<<dst[3] + dst[4] * gridX + dst[5] * gridY
		<<"\n ray[0] "<<(dst[3] - dst[0]).normal();*/
}

bool AFrustum::intersectPoint(const Vector3F & p) const
{
    cvx::Sphere ball;
    ball.set(p, .5f);
    return gjk::Intersect1<AFrustum, cvx::Sphere>::Evaluate(*this, ball);
}

const BoundingBox AFrustum::calculateBBox() const
{
    BoundingBox bx;
    for(int i=0; i < 8; i++) {
        bx.expandBy(m_v[i]);
    }
    return bx;
}

}