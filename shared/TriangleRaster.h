#ifndef TRIANGLERASTER_H
#define TRIANGLERASTER_H

/*
 *  TriangleRaster.h
 *  proxyPaint
 *
 *  Created by jian zhang on 3/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <Vector2F.h>
#include <Vector3F.h>
#include <Matrix44F.h>
#include <stdlib.h>
#include <PseudoNoise.h>
class TriangleRaster {
public:
	TriangleRaster();
	virtual ~TriangleRaster();
	
	static float barycentric_coord(float ax, float ay, float bx, float by, float x, float y);
	
	char create(const Vector3F & a, const Vector3F & b, const Vector3F & c);
	void gridSize(const float delta, int & num_grid_x, int & num_grid_y);
	void genSamples(const float delta, const int & num_grid_x, const int & num_grid_y, Vector3F *res, char *hits);

	char isPointWithin(const Vector3F &test, float &alpha, float & beta, float & gamma);
private:
	Matrix44F mat;
	Vector3F p[3];
	Vector3F q[3];
	Vector3F n[3];
	PseudoNoise pnoise;
	Vector2F bboxMin;
	Vector2F bboxMax;
	float f120, f201, f012;
	int _seed;
	char dice(int i, int j, float delta_x, float delta_y, float &alpha, float & beta, float & gamma );
	void eliminateSampleTooClose(float distance, int i, int j, int i1, int j1, int grid_x, int grid_y, Vector3F *res, char *hits);
};
#endif        //  #ifndef TRIANGLERASTER_H
