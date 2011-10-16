/*
 *  subdivision.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <QGLWidget>
#include "subdivision.h"

int getId(int i, int j, int w, int dim)
{
	return (j * w + i) * 3 + dim;
}

float getVec(float *d, int i, int dim)
{
	return d[i * 3 + dim];
}

float getVec(float *d, int i, int j, int w, int dim)
{
	return d[getId(i, j, w, dim)];
}

float averageOf(float *d, int a, int b, int dim)
{
	return (getVec(d, a, dim) + getVec(d, b, dim)) / 2.f;
}

void drawP(float *p, int i)
{
	glVertex3f(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
		
}

void drawF(float *p, int *c, int i)
{
	drawP(p, c[i*4]);
	drawP(p, c[i*4+1]);
	drawP(p, c[i*4+2]);
	drawP(p, c[i*4+3]);
}

float getNoise(float size, float mid = 1.f)
{
	random();
	return mid + size*(float(random() % 238) / 238.f - 0.5f);
}

int towpowerof(int exposure)
{
	int res = 1;
	for(int i= 0; i < exposure; i++)
		res *= 2;
	return res;
}

int faceCountAtLevel(int level)
{
	int w = towpowerof(level) + 2 + 2;
	int h = towpowerof(level) + 2;
	return w * h;
}

int nodeCountAtLevel(int level)
{
	int w = towpowerof(level) + 2 + 1 + 2;
	int h = towpowerof(level) + 2 + 1;
	return w * h;
}

int isCorner(int i, int j, int w)
{
	if(i == 1 && j == 0)
		return 0;
	else if(i == w-2 && j == 0)
		return 1;
	else if(i == w-2 && j == w-3)
		return 2;
	else if(i == 1 && j == w-3)
		return 3;
	return -1;
}

int vertexAt(int i, int j, int w)
{
	return j * w + i;
}

int faceAt(int i, int j, int w)
{
	return j * (w - 1) + i + 1;
}

void faceConnectionAtLevel(int* connection, int level, int *set)
{
	int patchWidth = towpowerof(level) + 1 + 2 + 2;
	int patchHeight = towpowerof(level) + 1 + 2;
	int faceId = 0;
	for(int j=0; j < patchHeight-1; j++)
	{
		for(int i=0; i < patchWidth-1; i++)
		{
			connection[faceId * 4] = vertexAt(i, j, patchWidth);
			connection[faceId * 4+1] = vertexAt(i + 1, j, patchWidth);
			connection[faceId * 4+2] = vertexAt(i + 1, j + 1, patchWidth);
			connection[faceId * 4+3] = vertexAt(i, j + 1, patchWidth);
			
			faceId++;
		}
	}

	int lastFace = towpowerof(level) + 1;
	int lastVert = patchWidth - 1;
	if(set[0] == 0)
	{
		int merge0 = faceAt(0, 1, patchWidth);
		int merge1 = faceAt(1, 0, patchWidth);
		connection[merge1*4] = connection[merge0*4];
	}
	if(set[0] == 2)
	{
		int merge0 = faceAt(0, 0, patchWidth);
		connection[merge0*4] = vertexAt(0, 0, patchWidth);
		connection[merge0*4+1] = vertexAt(2, 0, patchWidth);
		connection[merge0*4+2] = vertexAt(2, 1, patchWidth);
		connection[merge0*4+3] = vertexAt(1, 0, patchWidth);
		
		int merge1 = faceAt(-1, 0, patchWidth);
		connection[merge1*4] = vertexAt(0, 1, patchWidth);
		connection[merge1*4+1] = vertexAt(1, 0, patchWidth);
		connection[merge1*4+2] = vertexAt(2, 1, patchWidth);
		connection[merge1*4+3] = vertexAt(1, 1, patchWidth);

	}

	if(set[1] == 0)
	{
		int merge0 = faceAt(lastFace - 1, 0, patchWidth);
		int merge1 = faceAt(lastFace, 1, patchWidth);
		connection[merge1*4+1] = connection[merge0*4+1];
	}
	if(set[1] == 2)
	{
		int merge0 = faceAt(lastFace, 0, patchWidth);
		connection[merge0*4] = vertexAt(lastVert, 0, patchWidth);
		connection[merge0*4+1] = vertexAt(lastVert - 1, 0, patchWidth);
		connection[merge0*4+2] = vertexAt(lastVert - 2, 1, patchWidth);
		connection[merge0*4+3] = vertexAt(lastVert - 2, 0, patchWidth);
		
		int merge1 = faceAt(lastFace + 1, 0, patchWidth);
		connection[merge1*4] = vertexAt(lastVert, 1, patchWidth);
		connection[merge1*4+1] = vertexAt(lastVert - 1, 1, patchWidth);
		connection[merge1*4+2] = vertexAt(lastVert - 2, 1, patchWidth);
		connection[merge1*4+3] = vertexAt(lastVert - 1, 0, patchWidth);
	}
	
	if(set[2] == 0)
	{
		int merge0 = faceAt(lastFace, lastFace - 1, patchWidth);
		int merge1 = faceAt(lastFace - 1, lastFace, patchWidth);
		connection[merge1*4+2] = connection[merge0*4+2];
	}
	if(set[2] == 2)
	{
		int merge0 = faceAt(lastFace, lastFace, patchWidth);
		connection[merge0*4] = vertexAt(lastVert, lastVert - 2, patchWidth);
		connection[merge0*4+1] = vertexAt(lastVert - 2, lastVert - 2, patchWidth);
		connection[merge0*4+2] = vertexAt(lastVert - 2, lastVert - 3, patchWidth);
		connection[merge0*4+3] = vertexAt(lastVert - 1, lastVert - 2, patchWidth);
		
		int merge1 = faceAt(lastFace + 1, lastFace, patchWidth);
		connection[merge1*4] = vertexAt(lastVert, lastVert - 3, patchWidth);
		connection[merge1*4+1] = vertexAt(lastVert - 1, lastVert - 2, patchWidth);
		connection[merge1*4+2] = vertexAt(lastVert - 2, lastVert - 3, patchWidth);
		connection[merge1*4+3] = vertexAt(lastVert - 1, lastVert - 3, patchWidth);
	}
	
	
	if(set[3] == 0)
	{
		int merge0 = faceAt(1, lastFace, patchWidth);
		int merge1 = faceAt(0, lastFace - 1, patchWidth);
		connection[merge1*4+3] = connection[merge0*4+3];
	}
	if(set[3] == 2)
	{
		int merge0 = faceAt(0, lastFace, patchWidth);
		connection[merge0*4] = vertexAt(0, lastVert - 2, patchWidth);
		connection[merge0*4+1] = vertexAt(1, lastVert - 2, patchWidth);
		connection[merge0*4+2] = vertexAt(2, lastVert - 3, patchWidth);
		connection[merge0*4+3] = vertexAt(2, lastVert - 2, patchWidth);
		
		int merge1 = faceAt(-1, lastFace, patchWidth);
		connection[merge1*4] = vertexAt(0, lastVert - 3, patchWidth);
		connection[merge1*4+1] = vertexAt(1, lastVert - 3, patchWidth);
		connection[merge1*4+2] = vertexAt(2, lastVert - 3, patchWidth);
		connection[merge1*4+3] = vertexAt(1, lastVert - 2, patchWidth);
	}
	
}

void fillNodePositionAtLevel(float *p, int level)
{
	int patchWidth = towpowerof(level) + 1 + 2 + 2;
	int patchHeight = towpowerof(level) + 1 + 2;
	
	int nodeId = 0;
	for(int j=0; j < patchHeight; j++)
	{
		for(int i=0; i < patchWidth; i++)
		{
			p[nodeId * 3] = i + getNoise(.39f) - 3.5;
			
			if(i == 1 || i == 4)
				p[nodeId * 3+1] = -2.5f + getNoise(1.229f);
			else
				p[nodeId * 3+1] = -.5f + getNoise(1.229f);

			p[nodeId * 3+2] = j + getNoise(.29f) - 2.5;
			
			nodeId++;
		}
	}
	
	p[0] = p[3] + 0.5f;
	p[1] = p[4] + 0.25f;
	p[2] = p[5] - 0.15f;
		
	int extra = vertexAt(patchWidth - 1, 0, patchWidth);
	int corn = vertexAt(patchWidth - 2, 0, patchWidth);
	p[extra* 3] = p[corn * 3] - 0.43f;
	p[extra* 3 + 1] = p[corn * 3 + 1] - 0.03f;
	p[extra* 3 + 2] = p[corn * 3 + 2] - 0.13f;
	
	extra = vertexAt(0, 1, patchWidth);
	corn = vertexAt(0, 0, patchWidth);
	p[extra* 3] = p[corn * 3] - .73f;
	p[extra* 3 + 1] = p[corn * 3 + 1] - 0.01f;
	p[extra* 3 + 2] = p[corn * 3 + 2] + .8f;
	
	extra = vertexAt(patchWidth - 1, 1, patchWidth);
	corn = vertexAt(patchWidth-2, 0, patchWidth);
	p[extra* 3] = p[corn * 3] + .33f;
	p[extra* 3 + 1] = p[corn * 3 + 1] - 0.01f;
	p[extra* 3 + 2] = p[corn * 3 + 2] + .58f;
	
	extra = vertexAt(0, patchHeight - 2, patchWidth);
	corn = vertexAt(1, patchHeight - 1, patchWidth);
	p[extra* 3] = p[corn * 3] - 0.5;
	p[extra* 3 + 1] = p[corn * 3 + 1] ;
	p[extra* 3 + 2] = p[corn * 3 + 2] - 0.4;
	
	extra = vertexAt(0, patchHeight - 1, patchWidth);
	corn = vertexAt(1, patchHeight - 1, patchWidth);
	p[extra* 3] = p[corn * 3] + 0.3;
	p[extra* 3 + 1] = p[corn * 3 + 1] ;
	p[extra* 3 + 2] = p[corn * 3 + 2] + 0.4;
	
	extra = vertexAt(patchWidth - 1, patchHeight - 2, patchWidth);
	corn = vertexAt(patchWidth - 2, patchHeight - 1, patchWidth);
	p[extra* 3] = p[corn * 3] + 0.6;
	p[extra* 3 + 1] = p[corn * 3 + 1] ;
	p[extra* 3 + 2] = p[corn * 3 + 2] - 0.4;
	
	extra = vertexAt(patchWidth - 1, patchHeight - 1, patchWidth);
	corn = vertexAt(patchWidth - 2, patchHeight - 1, patchWidth);
	p[extra* 3] = p[corn * 3] - 0.3;
	p[extra* 3 + 1] = p[corn * 3 + 1] ;
	p[extra* 3 + 2] = p[corn * 3 + 2] + 0.4;
}

void updateFaceNode(float *p1, int node, float *p0, int face, int *c0)
{	
	int left = c0[face*4];
	int right = c0[face*4+1];
	int bottom = c0[face*4+2];
	int top = c0[face*4+3];
	p1[node * 3] = (getVec(p0, left, 0) + getVec(p0, right, 0) + getVec(p0, bottom, 0) + getVec(p0, top, 0)) / 4.f;
	p1[node * 3 + 1] = (getVec(p0, left, 1) + getVec(p0, right, 1) + getVec(p0, bottom, 1) + getVec(p0, top, 1)) / 4.f;
	p1[node * 3 + 2] = (getVec(p0, left, 2) + getVec(p0, right, 2) + getVec(p0, bottom, 2) + getVec(p0, top, 2)) / 4.f;
}

void weightedAverageEdge(float *p0, float *p1, int node, int corner0, int corner1, int corner2, int corner3, int edge0, int edge1)
{
	p1[node * 3] = ((getVec(p0, edge0, 0) + getVec(p0, edge1, 0)) * 6.f + getVec(p0, corner0, 0) + getVec(p0, corner1, 0) + getVec(p0, corner2, 0) + getVec(p0, corner3, 0)) / 16.f;
	p1[node * 3+1] = ((getVec(p0, edge0, 1) + getVec(p0, edge1, 1)) * 6.f + getVec(p0, corner0, 1) + getVec(p0, corner1, 1) + getVec(p0, corner2, 1) + getVec(p0, corner3, 1)) / 16.f;
	p1[node * 3+2] = ((getVec(p0, edge0, 2) + getVec(p0, edge1, 2)) * 6.f + getVec(p0, corner0, 2) + getVec(p0, corner1, 2) + getVec(p0, corner2, 2) + getVec(p0, corner3, 2)) / 16.f;
}

void updateEdgeCenterNode(float *p1, int node, float *p0, int edge0, int edge1)
{	
	p1[node * 3] = averageOf(p0, edge0, edge1, 0);
	p1[node * 3+1] = averageOf(p0, edge0, edge1, 1);
	p1[node * 3+2] = averageOf(p0, edge0, edge1, 2);
}

void updateBoundaryVertexNode(float *p1, int node, float *p0, int center, int corner0, int corner1)
{	
	p1[node * 3]     = getVec(p0, center, 0) * 0.75f + (getVec(p0, corner0, 0) + getVec(p0, corner1, 0)) * 0.125f;
	p1[node * 3 + 1] = getVec(p0, center, 1) * 0.75f + (getVec(p0, corner0, 1) + getVec(p0, corner1, 1)) * 0.125f;
	p1[node * 3 + 2] = getVec(p0, center, 2) * 0.75f + (getVec(p0, corner0, 2) + getVec(p0, corner1, 2)) * 0.125f;
}


void updateVertexNode(float *p1, int node, float *p0, int face0, int face1, int face2, int face3, int *c0)
{	
	int corner0 = c0[face0 * 4];
	int corner1 = c0[face1 * 4 + 1];
	int corner2 = c0[face2 * 4 + 2];
	int corner3 = c0[face3 * 4 + 3];
	int edge0 = c0[face0 * 4 + 3];
	int edge1 = c0[face1 * 4 + 0];
	int edge2 = c0[face2 * 4 + 1];
	int edge3 = c0[face3 * 4 + 2];
	int center = c0[face0 * 4 + 2];
	p1[node * 3] = (getVec(p0, center, 0) * 36.f + (getVec(p0, edge0, 0) + getVec(p0, edge1, 0) + getVec(p0, edge2, 0) + getVec(p0, edge3, 0)) * 6.f + (getVec(p0, corner0, 0) + getVec(p0, corner1, 0) + getVec(p0, corner2, 0) + getVec(p0, corner3, 0))) / 64.f;
	p1[node * 3 + 1] = (getVec(p0, center, 1) * 36.f + (getVec(p0, edge0, 1) + getVec(p0, edge1, 1) + getVec(p0, edge2, 1) + getVec(p0, edge3, 1)) * 6.f + (getVec(p0, corner0, 1) + getVec(p0, corner1, 1) + getVec(p0, corner2, 1) + getVec(p0, corner3, 1))) / 64.f;
	p1[node * 3 + 2] = (getVec(p0, center, 2) * 36.f + (getVec(p0, edge0, 2) + getVec(p0, edge1, 2) + getVec(p0, edge2, 2) + getVec(p0, edge3, 2)) * 6.f + (getVec(p0, corner0, 2) + getVec(p0, corner1, 2) + getVec(p0, corner2, 2) + getVec(p0, corner3, 2))) / 64.f;
}

void weightedAverageValence3(float *p1, int node, float *p0, int corner1, int corner2, int corner3, int edge1, int edge2, int edge3, int center)
{
	p1[node * 3] = (getVec(p0, center, 0) * 15.f + (getVec(p0, edge1, 0) + getVec(p0, edge2, 0) + getVec(p0, edge3, 0)) * 6.f + (getVec(p0, corner1, 0) + getVec(p0, corner2, 0) + getVec(p0, corner3, 0))) / 36.f;
	p1[node * 3 + 1] = (getVec(p0, center, 1) * 15.f + (getVec(p0, edge1, 1) + getVec(p0, edge2, 1) + getVec(p0, edge3, 1)) * 6.f + (getVec(p0, corner1, 1) + getVec(p0, corner2, 1) + getVec(p0, corner3, 1))) / 36.f;
	p1[node * 3 + 2] = (getVec(p0, center, 2) * 15.f + (getVec(p0, edge1, 2) + getVec(p0, edge2, 2) + getVec(p0, edge3, 2)) * 6.f + (getVec(p0, corner1, 2) + getVec(p0, corner2, 2) + getVec(p0, corner3, 2))) / 36.f;
}

void updateVetexNodeValence3(float *p1, int node, float *p0, int face0, int face1, int face2, int face3, int *c0, int corner)
{	
	int corner0 = c0[face0 * 4];
	int corner1 = c0[face1 * 4 + 1];
	int corner2 = c0[face2 * 4 + 2];
	int corner3 = c0[face3 * 4 + 3];
	int edge0 = c0[face0 * 4 + 3];
	int edge1 = c0[face1 * 4 + 0];
	int edge2 = c0[face2 * 4 + 1];
	int edge3 = c0[face3 * 4 + 2];
	int center = c0[face0 * 4 + 2];
	if(corner == 0)
	{
		center = c0[face1 * 4 + 3];
		weightedAverageValence3(p1, node, p0, corner1, corner2, corner3, edge1, edge2, edge3, center);
	}
	else if(corner == 1)
	{
		weightedAverageValence3(p1, node, p0, corner0, corner2, corner3, edge0, edge2, edge3, center);
	}
	else if(corner == 2)
	{
		weightedAverageValence3(p1, node, p0, corner0, corner1, corner3, edge0, edge1, edge3, center);
	}
	else if(corner == 3)
	{
		weightedAverageValence3(p1, node, p0, corner0, corner1, corner2, edge0, edge1, edge2, center);
	}
}

void updateVetexNodeValence5(float *p1, int node, float *p0, int face0, int face1, int face2, int face3, int face4, int *c0, int corner)
{	
	int corner0 = c0[face0 * 4 + 1];
	int corner1 = c0[face1 * 4 + 2];
	int corner2 = c0[face2 * 4 + 3];
	int corner3 = c0[face3 * 4];
	int corner4 = c0[face4 * 4];
	int edge0 = c0[face0 * 4];
	int edge1 = c0[face1 * 4 + 1];
	int edge2 = c0[face2 * 4 + 2];
	int edge3 = c0[face3 * 4 + 3];
	int edge4 = c0[face4 * 4 + 3];
	int center = c0[face0 * 4 + 3];
	
	if(corner == 1)
	{
		corner0 = c0[face0 * 4 + 2];
		corner1 = c0[face1 * 4 + 3];
		corner2 = c0[face2 * 4];
		edge0 = c0[face0 * 4 + 1];
		edge1 = c0[face1 * 4 + 2];
		edge2 = c0[face2 * 4 + 3];
		center = c0[face0 * 4];
	}
	else if(corner == 2)
	{
		corner0 = c0[face0 * 4 + 3];
		corner1 = c0[face1 * 4];
		corner2 = c0[face2 * 4 + 1];
		edge0 = c0[face0 * 4 + 2];
		edge1 = c0[face1 * 4 + 3];
		edge2 = c0[face2 * 4];
		center = c0[face0 * 4 + 1];
	}
	else if(corner == 3)
	{
		corner0 = c0[face0 * 4];
		corner1 = c0[face1 * 4 + 1];
		corner2 = c0[face2 * 4 + 2];
		edge0 = c0[face0 * 4 + 3];
		edge1 = c0[face1 * 4];
		edge2 = c0[face2 * 4 + 1];
		center = c0[face0 * 4 + 2];
	}
	
	p1[node * 3    ] = (getVec(p0, center, 0) * 65.f + (getVec(p0, edge0, 0) + getVec(p0, edge1, 0) + getVec(p0, edge2, 0) + getVec(p0, edge3, 0) + getVec(p0, edge4, 0)) * 6.f + (getVec(p0, corner0, 0) + getVec(p0, corner1, 0) + getVec(p0, corner2, 0) + getVec(p0, corner3, 0) + getVec(p0, corner4, 0))) / 100.f;
	p1[node * 3 + 1] = (getVec(p0, center, 1) * 65.f + (getVec(p0, edge0, 1) + getVec(p0, edge1, 1) + getVec(p0, edge2, 1) + getVec(p0, edge3, 1) + getVec(p0, edge4, 1)) * 6.f + (getVec(p0, corner0, 1) + getVec(p0, corner1, 1) + getVec(p0, corner2, 1) + getVec(p0, corner3, 1) + getVec(p0, corner4, 1))) / 100.f;
	p1[node * 3 + 2] = (getVec(p0, center, 2) * 65.f + (getVec(p0, edge0, 2) + getVec(p0, edge1, 2) + getVec(p0, edge2, 2) + getVec(p0, edge3, 2) + getVec(p0, edge4, 2)) * 6.f + (getVec(p0, corner0, 2) + getVec(p0, corner1, 2) + getVec(p0, corner2, 2) + getVec(p0, corner3, 2) + getVec(p0, corner4, 2))) / 100.f;
}

void fillBoundary(char *set)
{
	set[0]  = 1; set[1]  = 0; set[2]  = 0; set[3]  = 0; set[4]  = 1; 
	set[5]  = 1; set[6]  = 0; set[7]  = 1; set[8]  = 1; set[9]  = 1; 
	set[10] = 1; set[11] = 0; set[12] = 1; set[13] = 0; set[14] = 1;
}

void fillPatchSet(int *set)
{
	set[0] = 1; set[1] = 1;
	set[2] = 1; set[3] = 1;
}

void processBoundary(float *p1, float *p0, int *c0, int level, int *set, char *boundary)
{
	int patchWidth = towpowerof(level) + 1 + 2 + 2;
	int patchHeight = towpowerof(level) + 1 + 2;
	int faceWidth0 = towpowerof(level-1) + 2 + 2;
	int faceHeight0 = towpowerof(level-1) + 2;
	int face0, face1;

	if(!boundary[2])
	{
		for(int i= 1; i < patchWidth; i += 2)
		{
			face0 = faceWidth0 + i / 2 + 1;
			updateEdgeCenterNode(p1, patchWidth + i, p0, c0[face0*4], c0[face0*4+1]);
		}
		for(int i= 2; i < patchWidth; i += 2)
		{
			face0 = faceWidth0 + i / 2;
			face1 = faceWidth0 + i / 2 + 1;
			updateBoundaryVertexNode(p1, patchWidth + i, p0, c0[face0*4+1], c0[face0*4], c0[face1*4+1]);
		}
	}
	
	if(!boundary[8])
	{
		for(int j= 0; j < patchHeight; j += 2)
		{
			face0 = faceWidth0 * (j / 2) + (patchWidth - 3) / 2 ;
			updateEdgeCenterNode(p1, patchWidth * j + patchWidth - 3, p0, c0[face0*4 + 1], c0[face0*4+ 2]);
		}
		for(int j= 1; j < patchHeight; j += 2)
		{
			face0 = faceWidth0 * (j / 2) + (patchWidth - 3) / 2;
			face1 = faceWidth0 * (j / 2 + 1) + (patchWidth - 3) / 2;
			updateBoundaryVertexNode(p1, patchWidth * j + patchWidth - 3, p0, c0[face0*4+2], c0[face0*4+1], c0[face1*4+2]);
		}
	}
	
	if(!boundary[12])
	{
		for(int i= 1; i < patchWidth; i += 2)
		{
			face0 = faceWidth0 * (faceHeight0 - 2) + i / 2 + 1;
			updateEdgeCenterNode(p1, patchWidth * (patchHeight - 2) + i, p0, c0[face0*4 + 2], c0[face0*4+3]);
		}
		for(int i= 2; i < patchWidth; i += 2)
		{
			face0 = faceWidth0 * (faceHeight0 - 2) + i / 2;
			face1 = faceWidth0 * (faceHeight0 - 2) + i / 2 + 1;
			updateBoundaryVertexNode(p1, patchWidth * (patchHeight - 2) + i, p0, c0[face0*4+2], c0[face0*4 + 3], c0[face1*4+2]);
		}
	}
	
	if(!boundary[6])
	{
		for(int j= 0; j < patchHeight; j += 2)
		{
			face0 = faceWidth0 * (j / 2) + 2 ;
			updateEdgeCenterNode(p1, patchWidth * j + 2, p0, c0[face0*4 + 3], c0[face0*4]);
		}
		for(int j= 1; j < patchHeight; j += 2)
		{
			face0 = faceWidth0 * (j / 2) + 2;
			face1 = faceWidth0 * (j / 2 + 1) + 2;
			updateBoundaryVertexNode(p1, patchWidth * j + 2, p0, c0[face0*4 + 3], c0[face0*4], c0[face1*4+ 3]);
		}
	}
	if(!boundary[1])
	{
		updateEdgeCenterNode(p1, 2, p0, c0[2*4], c0[2*4+3]);
		updateEdgeCenterNode(p1, patchWidth + 1, p0, c0[(faceWidth0 + 1)*4], c0[(faceWidth0 + 1)*4+1]);
	}
	if(!boundary[3])
	{
		updateEdgeCenterNode(p1, patchWidth - 3, p0, c0[(faceWidth0 - 3)*4 + 1], c0[(faceWidth0 - 3)*4 + 2]);
		updateEdgeCenterNode(p1, patchWidth * 2 - 2, p0, c0[(faceWidth0 * 2 - 2)*4], c0[(faceWidth0 * 2 - 2)*4+1]);
	}
	if(!boundary[13])
	{
		updateEdgeCenterNode(p1, patchWidth * (patchHeight- 1) - 2, p0, c0[(faceWidth0 * (faceHeight0 - 1)- 2)*4 + 2], c0[(faceWidth0 * (faceHeight0 - 1)- 2)*4 + 3]);
		updateEdgeCenterNode(p1, patchWidth * patchHeight - 3, p0, c0[(faceWidth0 * faceHeight0 - 3)*4 + 1], c0[(faceWidth0 * faceHeight0 - 3)*4+2]);
	}
	if(!boundary[11])
	{
		updateEdgeCenterNode(p1, patchWidth * (patchHeight- 2) + 1, p0, c0[(faceWidth0 * (faceHeight0 - 2) + 1)*4 + 2], c0[(faceWidth0 * (faceHeight0 - 2) + 1)*4 + 3]);
		updateEdgeCenterNode(p1, patchWidth * (patchHeight- 1) + 2, p0, c0[(faceWidth0 * (faceHeight0 - 1) + 2)*4 + 3], c0[(faceWidth0 * (faceHeight0 - 1) + 2)*4]);
	}
	
	int end0 = -1, end1 = -1;
	if(!boundary[2])
	{
		end0 = c0[(faceWidth0 + 2) * 4 + 1];
	}
	if(!boundary[1])
	{
		if(end0 < 0) 
			end0 = c0[2 * 4];
	}
	if(!boundary[6])
	{
		if(end0 < 0) 
			end0 = c0[1 * 4 + 3];
		end1 = c0[(faceWidth0 + 2) * 4 + 3];
	}
	if(!boundary[1])
	{
		if(end1 < 0) 
			end1 = c0[(faceWidth0 + 1) * 4];
	}
	if(!boundary[2])
	{
		if(end1 < 0) 
			end1 = c0[1 * 4 + 1];
	}
	if(end1 > 0)
	{
		updateBoundaryVertexNode(p1, patchWidth + 2, p0, c0[(faceWidth0 + 2) * 4], end0, end1);
	}
	
	end0 = -1, end1 = -1;
	if(!boundary[8])
	{
		end0 = c0[(faceWidth0 * 2 - 3) * 4 + 2];
	}
	if(!boundary[3])
	{
		if(end0 < 0) 
			end0 = c0[(faceWidth0 * 2 - 2) * 4 + 1];
	}
	if(!boundary[2])
	{
		if(end0 < 0) 
			end0 = c0[(faceWidth0 - 2) * 4];
		end1 = c0[(faceWidth0 * 2 - 3) * 4];
	}
	if(!boundary[3])
	{
		if(end1 < 0) 
			end1 = c0[(faceWidth0 - 3) * 4 + 1];
	}
	if(!boundary[8])
	{
		if(end1 < 0) 
			end1 = c0[(faceWidth0 - 2) * 4 + 2];
	}
	if(end1 > 0)
	{
		updateBoundaryVertexNode(p1, patchWidth + patchWidth - 3, p0, c0[(faceWidth0 * 2 - 3) * 4 + 1], end0, end1);
	}
	
	end0 = -1, end1 = -1;
	if(!boundary[12])
	{
		end0 = c0[(faceWidth0 * (faceHeight0 - 1) - 3) * 4 + 3];
	}
	if(!boundary[13])
	{
		if(end0 < 0) 
			end0 = c0[(faceWidth0 * faceHeight0 - 3) * 4 + 2];
	}
	if(!boundary[8])
	{
		if(end0 < 0) 
			end0 = c0[(faceWidth0 * faceHeight0 - 2) * 4 + 1];
		end1 = c0[(faceWidth0 * (faceHeight0 - 1) - 3) * 4 + 1];
	}
	if(!boundary[13])
	{
		if(end1 < 0) 
			end1 = c0[(faceWidth0 * (faceHeight0 - 1) - 2) * 4 + 2];
	}
	if(!boundary[12])
	{
		if(end1 < 0) 
			end1 = c0[(faceWidth0 * faceHeight0 - 2) * 4 + 3];
	}
	if(end1 > 0)
	{
		updateBoundaryVertexNode(p1, patchWidth *(patchHeight - 1) - 3, p0, c0[(faceWidth0 * (faceHeight0 - 1) - 3) * 4 + 2], end0, end1);
	}
	
	end0 = -1, end1 = -1;
	if(!boundary[6])
	{
		end0 = c0[(faceWidth0 * (faceHeight0 - 2) + 2) * 4];
		printf("e0 %i\n", end0);
	}
	if(!boundary[11])
	{
		if(end0 < 0) 
			end0 = c0[(faceWidth0 * (faceHeight0 - 2) + 1) * 4 + 3];
	}
	if(!boundary[12])
	{
		if(end0 < 0) 
			end0 = c0[(faceWidth0 * (faceHeight0 - 1) + 1) * 4 + 2];
		end1 = c0[(faceWidth0 * (faceHeight0 - 2) + 2) * 4 + 2];
	}
	if(!boundary[11])
	{
		if(end1 < 0) 
			end1 = c0[(faceWidth0 * (faceHeight0 - 1) + 2) * 4 + 3];
	}
	if(!boundary[6])
	{
		if(end1 < 0) 
			end1 = c0[(faceWidth0 * (faceHeight0 - 1) + 1) * 4];
	}
	if(end1 > 0)
	{
		updateBoundaryVertexNode(p1, patchWidth * (patchHeight - 2) + 2, p0, c0[(faceWidth0 * (faceHeight0 - 2) + 2) * 4 + 3], end0, end1);
	}
}

void updateNodeAtLevel(float *p1, float *p0, int *c0, int level, int *set, char *boundary)
{
	int patchWidth = towpowerof(level) + 1 + 2 + 2;
	int patchHeight = towpowerof(level) + 1 + 2;
	int faceWidth0 = towpowerof(level-1) + 2 + 2;
	int faceHeight0 = towpowerof(level-1) + 2;
	int face0, face1;
	for(int j=0; j < patchHeight; j += 2)
	{
		for(int i= 1; i < patchWidth; i += 2)
		{
			updateFaceNode(p1, patchWidth * j + i, p0, j/2*faceWidth0 + i/2 + 1, c0);
		}
	}
	
	for(int j=0; j < patchHeight; j += 2)
	{
		for(int i= 2; i < patchWidth; i += 2)
		{
			face0 = j / 2 * faceWidth0 + i / 2;
			face1 = j / 2 * faceWidth0 + i / 2 + 1;
			weightedAverageEdge(p0, p1, patchWidth * j + i, c0[face0 * 4  + 3], c0[face0 * 4], c0[face1 * 4 + 1], c0[face1 * 4 + 2], c0[face0 * 4 + 1], c0[face0 * 4 + 2]);
		}
	}
	
	for(int j=1; j < patchHeight; j += 2)
	{
		for(int i= 1; i < patchWidth; i += 2)
		{
			face0 = j / 2 * faceWidth0 + i / 2 + 1;
			face1 = (j / 2 + 1) * faceWidth0 + i / 2 + 1;
			weightedAverageEdge(p0, p1, patchWidth * j + i, c0[face0 * 4], c0[face0 * 4 + 1], c0[face1 * 4 + 2], c0[face1 * 4 + 3], c0[face0 * 4 + 2], c0[face0 * 4 + 3]);
		}
	}
	
	for(int j=1; j < patchHeight; j += 2)
	{
		for(int i= 2; i < patchWidth; i += 2)
		{
			updateVertexNode(p1, patchWidth * j + i, p0, j/2*faceWidth0 + i/2, j/2*faceWidth0 + i/2 + 1, (j/2 + 1)*faceWidth0 + i/2 + 1, (j/2 + 1)*faceWidth0 + i/2, c0);
		}
	}
	
	if(set[0] == 0)
	{
		updateEdgeCenterNode(p1, patchWidth + 1, p0, c0[(faceWidth0 + 1)*4], c0[(faceWidth0 + 1)*4 + 1]);
		updateVetexNodeValence3(p1, patchWidth + 2, p0, 1, 2, faceWidth0 + 2, faceWidth0 + 1, c0, 0);
	}
	else if(set[0] == 2)
	{
		face0 = 1;
		face1 = 2;
		weightedAverageEdge(p0, p1, 2, c0[face0 * 4 + 3], c0[face0 * 4], c0[face1 * 4 + 1], c0[face1 * 4 + 2], c0[face0 * 4 + 1], c0[face0 * 4 + 2]);
		face0 = 0;
		face1 = faceWidth0 + 1;
		weightedAverageEdge(p0, p1, patchWidth + 1, c0[face0 * 4], c0[face0 * 4 + 1], c0[face1 * 4 + 2], c0[face1 * 4 + 3], c0[face0 * 4 + 2], c0[face0 * 4 + 3]);
		
		updateFaceNode(p1, 0, p0, 1, c0);
		updateFaceNode(p1, patchWidth, p0, 0, c0);
		updateEdgeCenterNode(p1, 1, p0, c0[1], c0[2]);
		
		updateVetexNodeValence5(p1, patchWidth + 2, p0, 2, faceWidth0 + 2, faceWidth0 + 1, 0, 1, c0, 0);
	}
	
	if(set[1] == 0)
	{
		updateEdgeCenterNode(p1, patchWidth - 3, p0, c0[(faceWidth0 - 3)*4 + 1], c0[(faceWidth0 - 3)*4 + 2]);
		updateVetexNodeValence3(p1, patchWidth + patchWidth - 3, p0, faceWidth0 - 3, faceWidth0 - 2, faceWidth0 + faceWidth0 - 2, faceWidth0 + faceWidth0 - 3, c0, 1);
	}
	else if(set[1] == 2)
	{
		face0 = faceWidth0 - 3;
		face1 = faceWidth0 - 2;
		
		weightedAverageEdge(p0, p1, patchWidth - 3, c0[face0 * 4 + 3], c0[face0 * 4], c0[face1 * 4], c0[face1 * 4 + 1], c0[face0 * 4 + 1], c0[face0 * 4 + 2]);
		
		face0 = faceWidth0 - 1;
		face1 = faceWidth0 * 2 - 2;
		weightedAverageEdge(p0, p1, patchWidth * 2 - 2, c0[face0 * 4 + 3], c0[face0 * 4], c0[face1 * 4 + 2], c0[face1 * 4 + 3], c0[face1 * 4], c0[face1 * 4 + 1]);
		
		updateFaceNode(p1, patchWidth - 1, p0, faceWidth0 - 2, c0);
		updateFaceNode(p1, patchWidth * 2 - 1, p0, faceWidth0 - 1, c0);
		
		face0 = faceWidth0 - 1;
		updateEdgeCenterNode(p1, patchWidth - 2, p0, c0[face0*4+2], c0[face0*4+3]);
		
		updateVetexNodeValence5(p1, patchWidth * 2 - 3, p0, faceWidth0 * 2 - 2, faceWidth0 * 2 - 3, faceWidth0 - 3, faceWidth0 - 2, faceWidth0 - 1,  c0, 1);
	}
	
	if(set[2] == 0)
	{
		updateEdgeCenterNode(p1, patchWidth * (patchHeight - 1) - 2, p0, c0[(faceWidth0 * (faceHeight0 - 1) - 2)*4 + 2], c0[(faceWidth0 * (faceHeight0 - 1) - 2)*4 + 3]);
		updateVetexNodeValence3(p1, patchWidth * (patchHeight - 1) - 3, p0, faceWidth0 * (faceHeight0 - 1) - 3, faceWidth0 * (faceHeight0 - 1) - 2, faceWidth0 * faceHeight0 - 2, faceWidth0 * faceHeight0 - 3, c0, 2);
	}
	else if(set[2] == 2)
	{
		face0 = faceWidth0 * faceHeight0 - 3;
		face1 = faceWidth0 * faceHeight0 - 2;
		weightedAverageEdge(p0, p1, patchWidth * patchHeight - 3, c0[face0 * 4 + 3], c0[face0 * 4], c0[face1 * 4 + 3], c0[face1 * 4], c0[face0 * 4 + 1], c0[face0 * 4 + 2]);
		face0 = faceWidth0 * (faceHeight0 - 1) - 2;
		face1 = faceWidth0 * faceHeight0 - 1;
		weightedAverageEdge(p0, p1, patchWidth * (patchHeight - 1) - 2, c0[face0 * 4], c0[face0 * 4 + 1], c0[face1 * 4], c0[face1 * 4 + 1], c0[face0 * 4 + 2], c0[face0 * 4 + 3]);
		updateFaceNode(p1, patchWidth * patchHeight - 1, p0, faceWidth0 * faceHeight0 - 2, c0);
		updateFaceNode(p1, patchWidth * (patchHeight -1 ) - 1, p0, faceWidth0 * faceHeight0 - 1, c0);
		
		face0 = faceWidth0 * faceHeight0 - 1;
		updateEdgeCenterNode(p1, patchWidth * patchHeight - 2, p0, c0[face0*4+1], c0[face0*4+2]);
		
		updateVetexNodeValence5(p1, patchWidth * (patchHeight -1) - 3, p0, faceWidth0 * faceHeight0 - 3, faceWidth0 * (faceHeight0 - 1) - 3, faceWidth0 * (faceHeight0 - 1) - 2, faceWidth0 * faceHeight0 - 1, faceWidth0 * faceHeight0 - 2, c0, 2);
	}
	
	if(set[3] == 0)
	{
		updateEdgeCenterNode(p1, patchWidth * (patchHeight - 1) + 2, p0, c0[(faceWidth0 * (faceHeight0 - 1) + 2)*4 + 3], c0[(faceWidth0 * (faceHeight0 - 1) + 2)*4]);
		updateVetexNodeValence3(p1, patchWidth * (patchHeight - 2) + 2, p0, faceWidth0 * (faceHeight0 - 2) + 1, faceWidth0 * (faceHeight0 - 2) + 2, faceWidth0 * (faceHeight0 - 1) + 2, faceWidth0 * (faceHeight0 - 1) + 1, c0, 3);
	}
	else if(set[3] == 2)
	{
		face0 = faceWidth0 * (faceHeight0 - 2) + 1;
		face1 = faceWidth0 * (faceHeight0 - 1);
		weightedAverageEdge(p0, p1, patchWidth * (patchHeight - 2) + 1, c0[face0 * 4], c0[face0 * 4 + 1], c0[face1 * 4 + 3], c0[face1 * 4], c0[face0 * 4 + 2], c0[face0 * 4 + 3]);
		face0 = faceWidth0 * (faceHeight0 - 1) + 1;
		face1 = faceWidth0 * (faceHeight0 - 1) + 2;
		weightedAverageEdge(p0, p1, patchWidth * (patchHeight - 1) + 2, c0[face0 * 4], c0[face0 * 4 + 1], c0[face1 * 4 + 1], c0[face1 * 4 + 2], c0[face0 * 4 + 2], c0[face0 * 4 + 3]);
		
		updateFaceNode(p1, patchWidth * (patchHeight - 2), p0, faceWidth0 * (faceHeight0 - 1), c0);
		updateFaceNode(p1, patchWidth * (patchHeight - 1), p0, faceWidth0 * (faceHeight0 - 1) + 1, c0);
		face0 = faceWidth0 * (faceHeight0 - 1);
		updateEdgeCenterNode(p1, patchWidth * (patchHeight - 1) + 1, p0, c0[face0*4+2], c0[face0*4+3]);
		updateVetexNodeValence5(p1, patchWidth * (patchHeight -2) + 2, p0, faceWidth0 * (faceHeight0 - 2) + 1, faceWidth0 * (faceHeight0 - 2) + 2, faceWidth0 * (faceHeight0 - 1) + 2, faceWidth0 * (faceHeight0 - 1) + 1, faceWidth0 * (faceHeight0 - 1), c0, 3);	
	}
	
	processBoundary(p1, p0, c0, level, set, boundary);	
	
}

void drawFace(float *p, int *c, int level)
{
	int w = towpowerof(level) + 2 + 2;
	int h = towpowerof(level) + 2;
	for(int j=1; j < h -1; j++)
	{
		for(int i = 2; i < w - 2; i++)
		{
			drawF(p, c, j * w + i);	
			
		}
	}
}

void drawPatch(float *p, int *c, int level, int *set)
{
	int w = towpowerof(level) + 2 + 2;
	int h = towpowerof(level) + 2;
	for(int j=0; j < h; j++)
	{
		for(int i = 1; i < w - 1; i++)
		{
			int onCorner = isCorner(i, j, w);
			if(onCorner < 0)
			{
				drawF(p, c, j * w + i);	
			}
			else
			{
				if(set[onCorner] > 0)
					drawF(p, c, j * w + i);	
				if(set[onCorner] > 1)
				{
					if(onCorner == 0)
						drawF(p, c, 0);
					else if(onCorner == 1)
						drawF(p, c, faceAt(w - 2, 0, w + 1));
					else if(onCorner == 2)
						drawF(p, c, faceAt(w - 2, h - 1, w + 1));
					else
						drawF(p, c, faceAt(-1, h - 1, w + 1));
				}
			}
		}
	}
}

Subdivision::Subdivision()
{
	_patch_set = new int[4];
	
	fillPatchSet(_patch_set);
	
	char *boundary = new char[15];
	
	fillBoundary(boundary);

	int _cage_num_face = faceCountAtLevel(0);
	_caga_connection = new int[4 * _cage_num_face];	
	_cage_vertices = new float[nodeCountAtLevel(0) * 3];
	
	faceConnectionAtLevel(_caga_connection, 0, _patch_set);
	
	fillNodePositionAtLevel(_cage_vertices, 0);

	int _bent_num_face = faceCountAtLevel(1);
	_bent_connection = new int[4 * _bent_num_face];
	_bent_vertices = new float[nodeCountAtLevel(1) * 3];
	
	faceConnectionAtLevel(_bent_connection, 1, _patch_set);
	
	updateNodeAtLevel(_bent_vertices, _cage_vertices, _caga_connection, 1, _patch_set, boundary);
	
	int l2_num_face = faceCountAtLevel(2);
	_l2_connection = new int[4 * l2_num_face];
	_l2_vertices = new float[nodeCountAtLevel(2) * 3];
	
	faceConnectionAtLevel(_l2_connection, 2, _patch_set);
	
	updateNodeAtLevel(_l2_vertices, _bent_vertices, _bent_connection, 2, _patch_set, boundary);
	
	int l3_num_face = faceCountAtLevel(3);
	_l3_connection = new int[4 * l3_num_face];
	_l3_vertices = new float[nodeCountAtLevel(3) * 3];
	
	faceConnectionAtLevel(_l3_connection, 3, _patch_set);
	
	updateNodeAtLevel(_l3_vertices, _l2_vertices, _l2_connection, 3, _patch_set, boundary);
	
	int l4_num_face = faceCountAtLevel(4);
	_l4_connection = new int[4 * l4_num_face];
	_l4_vertices = new float[nodeCountAtLevel(4) * 3];
	
	faceConnectionAtLevel(_l4_connection, 4, _patch_set);
	
	updateNodeAtLevel(_l4_vertices, _l3_vertices, _l3_connection,  4, _patch_set, boundary);
	
	int l5_num_face = faceCountAtLevel(5);
	_l5_connection = new int[4 * l5_num_face];
	_l5_vertices = new float[nodeCountAtLevel(5) * 3];
	
	faceConnectionAtLevel(_l5_connection, 5, _patch_set);
	
	updateNodeAtLevel(_l5_vertices, _l4_vertices, _l4_connection,  5, _patch_set, boundary);
}

void Subdivision::draw()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            
	glColor3f(1.f, 1.f, 1.f);
	glBegin(GL_QUADS);
	
	drawPatch(_cage_vertices, _caga_connection, 0, _patch_set);
	
	glColor3f(.5f, 1.f, 1.f);
	
	drawPatch(_bent_vertices, _bent_connection, 1, _patch_set);

	glColor3f(.1f, 5.f, 1.f);
	
	drawPatch(_l2_vertices, _l2_connection, 2, _patch_set);

	glColor3f(.1f, 1.f, .5f);
	
	drawFace(_l3_vertices, _l3_connection, 3);
	
	glColor3f(1.f, .5f, 0.f);

	drawFace(_l4_vertices, _l4_connection, 4);

	glColor3f(1.f, 0.f, 0.f);

	drawFace(_l5_vertices, _l5_connection, 5);
	
	glEnd();
	
}