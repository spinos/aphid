/*
 *  CubeMesh.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "GeodesicHemisphereMesh.h"
#include <cmath>
namespace aphid {

GeodesicHemisphereMesh::GeodesicHemisphereMesh(unsigned level)
{
    unsigned nv = (level + 1) * (level + 1) * 2;
    
    unsigned nf = level * level * 2 * 2;
    
    createVertices(nv);
	createIndices(nf * 3);

	Vector3F * p = vertices();
	
	unsigned * idx = indices();
	
	unsigned currentidx = 0;
	unsigned currentver = 0;
	
	Vector3F a(0.f, 1.f, 0.f);
	Vector3F b(-1.f, 0.001f, 0.f);
	Vector3F c(0.f, 0.001f, 1.f);
	Vector3F d(1.f, 0.001f, 0.f);
	Vector3F e(0.f, 0.001f, -1.f);
	
	subdivide(level, currentver, currentidx, p, idx, a, b, c, d);
	subdivide(level, currentver, currentidx, p, idx, a, d, e, b);

    printf("vertices count: %d\n", currentver);
    printf("indices count: %d\n", currentidx);	
}

GeodesicHemisphereMesh::~GeodesicHemisphereMesh() {}

void GeodesicHemisphereMesh::subdivide(unsigned level, unsigned & currentVertex, unsigned & currentIndex, Vector3F * p, unsigned * idx, Vector3F a, Vector3F b, Vector3F c, Vector3F d)
{
    unsigned offset = currentVertex;
    Vector3F delta_ab = (b - a) / (float)level;
    Vector3F delta_bc = (c - b) / (float)level;
    Vector3F delta_ad = (d - a) / (float)level;
    for(unsigned j = 0; j <= level; j++)
    {
        Vector3F row = a + delta_ab * (float)j;
        p[currentVertex] = row;           
        currentVertex++; 
        for(unsigned i = 1; i <= level; i++)
        {
            if(i <= j) row += delta_bc;
            else row += delta_ad;
            
            p[currentVertex] = row;           
            currentVertex++; 
        }
    }
    
    for(unsigned j = 0; j < level; j++)
    {
        for(unsigned i = 0; i < level; i++)
        {
            idx[currentIndex] = j * (level + 1) + i + offset;           
            currentIndex++;
            
            idx[currentIndex] = (j + 1) * (level + 1) + (i + 1) + offset;           
            currentIndex++;
            
            idx[currentIndex] = (j + 1) * (level + 1) + i + offset;           
            currentIndex++;
            
            idx[currentIndex] = j * (level + 1) + i + offset;           
            currentIndex++;
            
            idx[currentIndex] = j * (level + 1) + i + 1 + offset;           
            currentIndex++;
            
            idx[currentIndex] = (j + 1) * (level + 1) + (i + 1) + offset;           
            currentIndex++;
        }
    }
}

}