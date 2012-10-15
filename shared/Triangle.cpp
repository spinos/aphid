/*
 *  Triangle.cpp
 *  arum
 *
 *  Created by jian zhang on 9/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "Triangle.h"

//#include <maya/MGlobal.h>

void project_point_on_line(const Vector3F & p, const Vector3F & lineDir, const Vector3F & onLine, float & l)
{
	Vector3F d = p - onLine;
	l = d.dot(lineDir);
}

char ray_plane_intersect(const Vector3F & origin, const Vector3F & ray, const Vector3F & onPlane, const Vector3F & planeNormal, float & t, Vector3F & hit)
{
	float rdotn = ray.dot(planeNormal);
	
	if(rdotn > -10e-5f && rdotn < 10e-5f) return 0;
	
	t = (onPlane.dot(planeNormal) - origin.dot(planeNormal)) / rdotn;
	
	hit = origin + ray * t;
	
	return t >= 0.f;
}

Triangle::Triangle(const Vector3F& a, const Vector3F& b, const Vector3F& c, const Vector3F& n) 
{
	p0 = a;
	p1 = b;
	p2 = c;
	nor = n;
	
	//MGlobal::displayInfo(MString("n ") + n.x + " " + n.y + " " + n.z);
	
}

Triangle::~Triangle() {}

char Triangle::intersects(const Triangle * another) const
{
	float ndotn = nor.dot(another->nor);
	if(ndotn > 0.999999f || ndotn < -0.999999f) return 0;
	
	float t0, t1, t2;
	Vector3F ha, hb, hc;
	
	ray_plane_intersect(another->p0, nor, p0, nor, t0, ha);
	ray_plane_intersect(another->p1, nor, p0, nor, t1, hb);
	ray_plane_intersect(another->p2, nor, p0, nor, t2, hc);
	
	if(t0 < 0.f &&  t1 < 0.f && t2 < 0.f) return 0;
	if(t0 > 0.f &&  t1 > 0.f && t2 > 0.f) return 0;
	
	ray_plane_intersect(p0, another->nor, another->p0, another->nor, t0, ha);
	ray_plane_intersect(p1, another->nor, another->p0, another->nor, t1, hb);
	ray_plane_intersect(p2, another->nor, another->p0, another->nor, t2, hc);
	
	if(t0 < 0.f &&  t1 < 0.f && t2 < 0.f) return 0;
	if(t0 > 0.f &&  t1 > 0.f && t2 > 0.f) return 0;
	
	Vector3F o, ra, rb;
	if(t0 * t1 < 0.f && t0 * t2 < 0.f) {
		ra = p1 - p0;
		rb = p2 - p0;
		o = p0;
	}
	else if(t1 * t2 < 0.f && t1 * t0 < 0.f) {
		ra = p2 - p1;
		rb = p0 - p1;
		o = p1;
	}
	else {
		ra = p0 - p2;
		rb = p1 - p2;
		o = p2;
	}
	
	ray_plane_intersect(o, ra, another->p0, another->nor, t0, ha);
	ray_plane_intersect(o, rb, another->p0, another->nor, t1, hb);
	
	//MGlobal::displayInfo(MString("t ")+t0 + " " + t1);

	Vector3F D = hb - ha;
	float L = D.length();
	D.normalize();
	
	Vector3F perpe = another->nor.cross(D);
	
	Vector3F pa, pb, pc;
	float ta, tb, tc;
	
	ray_plane_intersect(another->p0, perpe, ha, perpe, ta, pa);
	ray_plane_intersect(another->p1, perpe, ha, perpe, tb, pb);
	ray_plane_intersect(another->p2, perpe, ha, perpe, tc, pc);
	
	if(ta * tb < 0.f && ta * tc < 0.f) {
		ra = another->p1 - another->p0;
		rb = another->p2 - another->p0;
		o = another->p0;
	}
	else if(tb * tc < 0.f && tb * ta < 0.f) {
		ra = another->p2 - another->p1;
		rb = another->p0 - another->p1;
		o = another->p1;
	}
	else {
		ra = another->p0 - another->p2;
		rb = another->p1 - another->p2;
		o = another->p2;
	}
	
	ray_plane_intersect(o, ra, ha, perpe, ta, pa);
	ray_plane_intersect(o, rb, ha, perpe, tb, pb);

	Vector3F da = pa - ha;
	Vector3F db = pb - ha;
	
	if(da.dot(D) < 0.f && db.dot(D) < 0.f) return 0;
	
	if(da.dot(D) > 0.f && db.dot(D) > 0.f) {
		if(da.length() > L && db.length() > L)
			return 0;
	}
	
	return 1;
}

// x = o + R t 
// x . n + d = 0
// d = - p0 . n	
// (o + R t) . n = p0 . n
// R t . n = p0 . n - o . n 
// t = (p0 . n - o . n ) / R . n


char Triangle::intersects(const Vector3F& origin, const Vector3F& ray, float maxDistance, Vector3F &position, Vector3F &normal) const
{
	float ddotn = ray.dot(nor);
		
	if(ddotn < 10e-5 && ddotn > -10e-5) return 0;
	
	float t = (p0.dot(nor) - origin.dot(nor)) / ddotn;
	
	if(t < 0.f || t > maxDistance) return 0;
	
	Vector3F onplane = origin + ray * t;
	Vector3F e01 = p1 - p0;
	Vector3F x0 = onplane - p0;
	if(e01.cross(x0).dot(nor) < 0.f) return 0;
	
	Vector3F e12 = p2 - p1;
	Vector3F x1 = onplane - p1;
	if(e12.cross(x1).dot(nor) < 0.f) return 0;
	
	Vector3F e20 = p0 - p2;
	Vector3F x2 = onplane - p2;
	if(e20.cross(x2).dot(nor) < 0.f) return 0;
	
	position = onplane;
	normal = nor;
	
	return 1;
}

char Triangle::frontIntersects(const Vector3F& origin, const Vector3F& ray, float maxDistance, Vector3F &position, Vector3F &normal) const
{
	if(ray.dot(nor) > 0.f) return 0;
	return intersects(origin, ray, maxDistance, position, normal);
}

char Triangle::backIntersects(const Vector3F& origin, const Vector3F& ray, float maxDistance, Vector3F &position, Vector3F &normal) const
{
	if(ray.dot(nor) < 0.f) return 0;
	return intersects(origin, ray, maxDistance, position, normal);
}

char Triangle::closestHit(const Vector3F& origin, Vector3F& dest, float maxDistance) const
{
	float t = (p0.dot(nor) - origin.dot(nor));
	
	if(t > maxDistance || t < -maxDistance) return 0;
	
	Vector3F onplane = origin + nor * t;
	Vector3F e01 = p1 - p0;
	Vector3F x0 = onplane - p0;
	if(e01.cross(x0).dot(nor) < 0.f) return 0;
	
	Vector3F e12 = p2 - p1;
	Vector3F x1 = onplane - p1;
	if(e12.cross(x1).dot(nor) < 0.f) return 0;
	
	Vector3F e20 = p0 - p2;
	Vector3F x2 = onplane - p2;
	if(e20.cross(x2).dot(nor) < 0.f) return 0;
	
	dest = onplane;
	
	return 1;
}

Vector3F Triangle::center() const
{
	return (p0 + p1 + p2)/3.f;
}

Vector3F Triangle::randomOnPlane() const
{
	float a = float(rand()%1349137)/1349137.f;
	float b = (float(rand()%1349191)/1349191.f)*(1.f - a);
	float c = 1.f - a - b;
	return p0 * a + p1 * b + p2 * c;
}

Vector3F Triangle::normal() const
{
	return nor;
}

void Triangle::name() const
{
	printf("tri");
}
