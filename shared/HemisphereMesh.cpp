/*
 *  CubeMesh.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "HemisphereMesh.h"
#include <cmath>

HemisphereMesh::HemisphereMesh(unsigned grid_theta, unsigned grid_phi)
{	
	_gridTheta = grid_theta;
	_gridPhi = grid_phi;
	
	createVertices(grid_theta * grid_phi);
	createIndices((grid_theta - 1) * grid_phi * 2 * 3);

	// Vector3F * p = vertices();
	// 
	// float delta_theta = 3.1415927f * 0.5f /(float)grid_theta;
	// float delta_phi = 3.1415927f * 2.f /(float)grid_phi;
	// 
	// for(unsigned j = 0; j < grid_theta; j++)
	// {
		// for(unsigned i = 0; i < grid_phi; i++)
		// {
			// p[j * grid_phi + i].x = sin(delta_theta * j) * cos(delta_phi * i);
			// p[j * grid_phi + i].y = sin(delta_theta * j) * sin(delta_phi * i);
			// p[j * grid_phi + i].z = cos(delta_theta * j);
		// }
	// }
	
	unsigned * idx = indices();
	
	for(unsigned j = 0; j < grid_theta - 1; j++)
	{
		for(unsigned i = 0; i < grid_phi; i++)
		{
			unsigned g = j * grid_phi + i;
			
			unsigned i1 = i + 1;
			if(i == grid_phi - 1) i1 = 0;
			
			idx[g * 6] = j * grid_phi + i;
			idx[g * 6 + 1] = (j + 1) * grid_phi + i;
			idx[g * 6 + 2] = (j + 1) * grid_phi + i1;
			idx[g * 6 + 3] = (j + 1) * grid_phi + i1;
			idx[g * 6 + 4] = j * grid_phi + i1;
			idx[g * 6 + 5] = j * grid_phi + i;
		}
	}
}

HemisphereMesh::~HemisphereMesh() {}

unsigned HemisphereMesh::getGridTheta() const
{
	return _gridTheta;
}

unsigned HemisphereMesh::getGridPhi() const
{
	return _gridPhi;
}
