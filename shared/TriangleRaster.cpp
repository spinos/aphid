/*
 *  TriangleRaster.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 3/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <math.h>
#include "TriangleRaster.h"

namespace aphid {

TriangleRaster::TriangleRaster() {}
TriangleRaster::~TriangleRaster() {}

float TriangleRaster::barycentric_coord(float ax, float ay, float bx, float by, float x, float y)
{
	return (ay - by)*x + (bx - ax)*y +ax*by - bx*ay;
}

char TriangleRaster::create(const Vector3F & a, const Vector3F & b, const Vector3F & c)
{
	p[0] = a;
	p[1] = b;
	p[2] = c;
	
	Vector3F v = b - a;
	
	if(v.length() < 1e-3f)
		return 0;
		
	v = c - b;
	if(v.length() < 1e-3f)
		return 0;
		
	v = a - c;
	if(v.length() < 1e-3f)
		return 0;
		
	Vector3F side = b - a; side.normalize();
	Vector3F up = c - a; up.normalize();
	Vector3F front = side.cross(up); front.normalize();
	up = front.cross(side); up.normalize();
	
	
	mat.setIdentity();
	mat.setOrientations(side, up, front);
	mat.setTranslation(a);
	mat.inverse();
	
	q[0] = mat.transform(p[0]);	
	q[1] = mat.transform(p[1]);	
	q[2] = mat.transform(p[2]);	
	
	f120 = barycentric_coord(q[1].x, q[1].y, q[2].x, q[2].y, q[0].x, q[0].y);
	f201 = barycentric_coord(q[2].x, q[2].y, q[0].x, q[0].y, q[1].x, q[1].y);
	f012 = barycentric_coord(q[0].x, q[0].y, q[1].x, q[1].y, q[2].x, q[2].y);
	
	n[0] = side;
	n[1] = up;
	n[2] = front;
	
	bboxMax.x = -1e6f;
	bboxMax.y = -1e6f;
	bboxMin.x = 1e6f;
	bboxMin.y = 1e6f;
	for(int i=0; i<3; i++) {
		if(q[i].x > bboxMax.x) bboxMax.x = q[i].x;
		if(q[i].y > bboxMax.y) bboxMax.y = q[i].y;
		if(q[i].x < bboxMin.x) bboxMin.x = q[i].x;
		if(q[i].y < bboxMin.y) bboxMin.y = q[i].y;
	}
	
	return 1;
}

void TriangleRaster::gridSize(const float delta, int & num_grid_x, int & num_grid_y)
{
	num_grid_x = (bboxMax.x - bboxMin.x)/delta + 1;
	num_grid_y = (bboxMax.y - bboxMin.y)/delta + 1;
}

void TriangleRaster::genSamples(const float delta, const int & num_grid_x, const int & num_grid_y, Vector3F *res, char *hits)
{	
	_seed = rand() % 92043;
	if(num_grid_y == 1 && num_grid_x == 1) {
		hits[0] = 0;
		float probablity = 1.f / (delta / (bboxMax.x - bboxMin.x)) / (delta / (bboxMax.y - bboxMin.y));
		float r = pnoise.rfloat(_seed);
		_seed++;
		if(r < probablity) {
			res[0] = p[0] * 0.3333f + p[1] * 0.3333f + p[2] * 0.3333f;
			hits[0] = 1;
		}
		return;
	}
	
	float alpha, beta, gamma;
	
	if(num_grid_y == 1) {
		float probablity = 1.f / (delta / (bboxMax.y - bboxMin.y));
		
		for(int i=0; i<num_grid_x; i++) {
			hits[i] = 0;
			float r = pnoise.rfloat(_seed);
			_seed++;
			if(r > probablity) 
				continue;
				
			if(!dice(i, 0, delta, bboxMax.y - bboxMin.y, alpha, beta, gamma ))
				continue;
				
			res[i] = p[0] * alpha + p[1] * beta + p[2] * gamma;
			hits[i] = 1;
		}
		
		return;
	}
	
	if(num_grid_x == 1) {
		float probablity = 1.f / (delta / (bboxMax.x - bboxMin.x));
		
		for(int i=0; i<num_grid_y; i++) {
			hits[i] = 0;
			float r = pnoise.rfloat(_seed);
			_seed++;
			if(r > probablity) 
				continue;
				
			if(!dice(0, i, bboxMax.x - bboxMin.x, delta,  alpha, beta, gamma ))
				continue;
				
			res[i] = p[0] * alpha + p[1] * beta + p[2] * gamma;
			hits[i] = 1;
		}
		
		return;
	}

	for(int j=0; j<num_grid_y; j++)
	{
		for(int i=0; i<num_grid_x; i++)
		{
			hits[j * num_grid_x + i] = 0;
			if(!dice(i, j, delta, delta, alpha, beta, gamma ))
				continue;
			res[j * num_grid_x + i] = p[0] * alpha + p[1] * beta + p[2] * gamma;
			hits[j * num_grid_x + i] = 1;
		}
	}
	
	float rdelta = delta * 0.51f;
	
	for(int j=0; j<num_grid_y; j++)
	{
		for(int i=0; i<num_grid_x; i++)
		{
			eliminateSampleTooClose(rdelta, i, j, i-1, j-1, num_grid_x, num_grid_y, res, hits);
			eliminateSampleTooClose(rdelta, i, j, i  , j-1, num_grid_x, num_grid_y, res, hits);
			eliminateSampleTooClose(rdelta, i, j, i+1, j-1, num_grid_x, num_grid_y, res, hits);
			eliminateSampleTooClose(rdelta, i, j, i-1, j, num_grid_x, num_grid_y, res, hits);
			eliminateSampleTooClose(rdelta, i, j, i+1, j, num_grid_x, num_grid_y, res, hits);
			eliminateSampleTooClose(rdelta, i, j, i-1, j+1, num_grid_x, num_grid_y, res, hits);
			eliminateSampleTooClose(rdelta, i, j, i  , j+1, num_grid_x, num_grid_y, res, hits);
			eliminateSampleTooClose(rdelta, i, j, i+1, j+1, num_grid_x, num_grid_y, res, hits);
		}
	}
}

void TriangleRaster::eliminateSampleTooClose(float distance, int i, int j, int i1, int j1, int grid_x, int grid_y, Vector3F *res, char *hits)
{
	if(i1 < 0 || j1 < 0)
		return;
	if(i1 >= grid_x || j1 >= grid_y)
		return;
		
	if(!hits[j1 * grid_x + i1])
		return;
		
	Vector3F dif = res[j * grid_x + i] - res[j1 * grid_x + i1];
	if(dif.length() < distance)
		hits[j * grid_x + i] = 0;
}

char TriangleRaster::dice(int i, int j, float delta_x, float delta_y, float &alpha, float & beta, float & gamma )
{
	for(int k=0; k < 15; k++) {
		float x = bboxMin.x + delta_x*(i+ 0.5f + (pnoise.rfloat(_seed) - 0.5f) * .79f); _seed++;
		float y = bboxMin.y + delta_y*(j+ 0.5f + (pnoise.rfloat(_seed) - 0.5f) * .79f); _seed++;
		alpha = barycentric_coord(q[1].x, q[1].y, q[2].x, q[2].y, x, y)/f120;
		beta = barycentric_coord(q[2].x, q[2].y, q[0].x, q[0].y, x, y)/f201;
		gamma = barycentric_coord(q[0].x, q[0].y, q[1].x, q[1].y, x, y)/f012;
		
		if(alpha>0 && alpha<1 && beta>0 && beta<1 && gamma>0 && gamma<1)
			return 1;
	}
	return 0;
}

char TriangleRaster::isPointWithin(const Vector3F &test, float &alpha, float & beta, float & gamma)
{
	Vector3F local = mat.transform(test);
	alpha = barycentric_coord(q[1].x, q[1].y, q[2].x, q[2].y, local.x, local.y)/f120;
	beta = barycentric_coord(q[2].x, q[2].y, q[0].x, q[0].y, local.x, local.y)/f201;
	gamma = barycentric_coord(q[0].x, q[0].y, q[1].x, q[1].y, local.x, local.y)/f012;
	
	if(alpha>0 && alpha<1 && beta>0 && beta<1 && gamma>0 && gamma<1)
		return 1;
			
	return 0;
}

}