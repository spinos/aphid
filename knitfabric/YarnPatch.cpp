/*
 *  YarnPatch.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 6/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "YarnPatch.h"
#include <BiLinearInterpolate.h>

YarnPatch::YarnPatch() 
{
	m_numWaleEdges = 0;
	m_numWaleGrid = 4;
	m_numCourseGrid[0] = 4;
	m_numCourseGrid[1] = 4;
	m_hasTessellation = 0;
}

void YarnPatch::setQuadVertices(unsigned *v)
{
	m_quadVertices = v;
}

void YarnPatch::findWaleEdge(unsigned v0, unsigned v1, int bothSide)
{
	short i, j;
	for(i = 0; i < 4; i++) {
		j = i + 1;
		if(j == 4) j = 0;
		if(m_quadVertices[i] == v0 && m_quadVertices[j] == v1) {
			setWaleEdge(i, j);
			if(bothSide == 1) setSecondWaleEdge(j, 1);
		}
	}
	for(i = 3; i >= 0; i--) {
		j = i - 1;
		if(j < 0) j = 3;
		if(m_quadVertices[i] == v0 && m_quadVertices[j] == v1) {
			setWaleEdge(i, j);
			if(bothSide == 1) setSecondWaleEdge(j, 0);
		}
	}
}

void YarnPatch::setWaleEdge(short i, short j)
{
	if(m_numWaleEdges == 2) m_numWaleEdges = 0;
	m_waleVertices[m_numWaleEdges * 2] = i;
	m_waleVertices[m_numWaleEdges * 2 + 1] = j;
	m_numWaleEdges++;
}

void YarnPatch::setSecondWaleEdge(short j, char dir)
{
	short a, b;
	if(dir) {
		a = j + 1; 
		if(a == 4) a = 0;
		b = a + 1;
		if(b == 4) b = 0;
	}
	else {
		a = j - 1;
		if(a < 0) a = 3;
		b = a - 1;
		if(b < 0) b = 3;
	}
	setWaleEdge(b, a);
}

void YarnPatch::waleEdges(short &n, unsigned * v) const
{
	if(m_numWaleEdges < 1) return;
	n = m_numWaleEdges;
	v[0] = m_quadVertices[m_waleVertices[0]];
	v[1] = m_quadVertices[m_waleVertices[1]];
	if(m_numWaleEdges < 2) return;
	v[2] = m_quadVertices[m_waleVertices[2]];
	v[3] = m_quadVertices[m_waleVertices[3]];
}

char YarnPatch::hasWaleEdges() const
{
	return (m_numWaleEdges == 2);
}

char YarnPatch::hasTessellation() const
{
	return m_hasTessellation;
}

char YarnPatch::verifyNumGrid()
{
    if(m_numWaleGrid < 2) return 0;
	if(m_numCourseGrid[0] < 2 || m_numCourseGrid[1] < 2) return 0;
	
	if(m_numCourseGrid[0] != m_numCourseGrid[1]) {
		short delta = m_numCourseGrid[0] - m_numCourseGrid[1];
		if(delta < 0) delta = -delta;
		if(delta >= m_numWaleGrid) 
			return 0;
	}
	return 1;
}
#include <iostream>
void YarnPatch::tessellate()
{
	if(!hasWaleEdges()) return;
	if(m_hasTessellation) BaseMesh::cleanup();
	m_hasTessellation = 0;
	if(!verifyNumGrid()) return;
	
	if(isTriangle()) tessellateTriangle();
	else tessellateQuad();
	
}

void YarnPatch::fillP(const short & nrow, const short & ncol0, const short & ncol1, const short & colDir)
{
	unsigned nv = 0;
	short rowEnd = ncol0;
	short colStep = colDir;
	short nextRowEnd = rowEnd;
	
	for(short j = 0; j < nrow; j++) {
		rowDifference(colStep, nextRowEnd, ncol1, j);
		nv += rowEnd;
		rowEnd = nextRowEnd;
	}
	
	createVertices(nv);
	
	float cu[4], cv[4];
	getCornerUV(m_waleVertices[0], cu[0], cv[0]);
	getCornerUV(m_waleVertices[1], cu[3], cv[3]);
	getCornerUV(m_waleVertices[2], cu[1], cv[1]);
	getCornerUV(m_waleVertices[3], cu[2], cv[2]);
	
	BiLinearInterpolate bilinear;
	Vector3F * pos = vertices();
			
	float u, v;
	const float dv = 1.f / (nrow - 1);
	float du;
	float pau, pav;
	
	nv = 0;
	rowEnd = ncol0;
	colStep = colDir;
	nextRowEnd = rowEnd;
	for(short j = 0; j < nrow; j++) {
		v = dv * j;
		
		rowDifference(colStep, nextRowEnd, ncol1, j);
		
		du = 1.f / (rowEnd - 1);
		for(short i = 0; i < rowEnd; i++) {
			
			u = du * i;
			pau = bilinear.interpolate(u, v, cu);
			pav = bilinear.interpolate(u, v, cv);
			
			//printf("%f %f- ", pau, pav);
			evaluateSurfacePosition(pau, pav, &pos[nv]);
			
			nv++;
		}

		rowEnd = nextRowEnd;
	}
}

void YarnPatch::fillF(const short & nrow, const short & ncol0, const short & ncol1, const short & colDir)
{
	unsigned nf = 0;
	short rowEnd = ncol0;
	short colStep = colDir;
	short nextRowEnd = rowEnd;
	short rowIncrease = 0;
	for(short j = 0; j < nrow - 1; j++) {
		rowIncrease = rowDifference(colStep, nextRowEnd, ncol1, j);
		if(rowIncrease < 0) rowEnd = nextRowEnd;
		nf += rowEnd - 1;
		rowEnd = nextRowEnd;
	}
	
	createPolygonCounts(nf);
	
	nf = 0;
	rowEnd = ncol0;
	colStep = colDir;
	
	unsigned *fc = polygonCounts();
	unsigned nfv = 0;
	rowIncrease = 0;
	nextRowEnd = rowEnd;
	for(short j = 0; j < nrow - 1; j++) {
		rowIncrease = rowDifference(colStep, nextRowEnd, ncol1, j);
		if(rowIncrease < 0) rowEnd = nextRowEnd;
		for(short i = 0; i < rowEnd - 1; i++) {
			if(i == 0) {
				if(rowIncrease != 0) {
					nfv += 5;
					fc[nf] = 5;
				}
				else {
					nfv += 4;
					fc[nf] = 4;
				}
			}
			else {
				nfv += 4;
				fc[nf] = 4;
			}
			nf++;
		}
		rowEnd = nextRowEnd;
		
	}
	
	createPolygonIndices(nfv);
	
	rowEnd = ncol0;
	colStep = colDir;
	rowIncrease = 0;
	unsigned currentRowStart = 0;
	unsigned nextRowStart = currentRowStart + rowEnd;
	nextRowEnd = rowEnd;
	unsigned currentIdx = 0;
	unsigned * fv = polygonIndices();
	for(short j = 0; j < nrow - 1; j++) {
		rowIncrease = rowDifference(colStep, nextRowEnd, ncol1, j);
		//printf("row %i: %i, %i\n", j, currentRowStart, nextRowStart-1);
		for(short i = 0; i < rowEnd - 1; i++) {
			if(i == 0) {
				if(rowIncrease > 0) {
					fv[currentIdx] = currentRowStart;
					fv[currentIdx + 1] = currentRowStart + 1;
					fv[currentIdx + 2] = nextRowStart + 2;
					fv[currentIdx + 3] = nextRowStart + 1;
					fv[currentIdx + 4] = nextRowStart;
					//printf("increase %i %i %i %i %i\n", fv[currentIdx], fv[currentIdx + 1], fv[currentIdx + 2], fv[currentIdx + 3], fv[currentIdx + 4]);
					currentIdx += 5;
				}
				else if(rowIncrease < 0) {
					fv[currentIdx] = currentRowStart;
					fv[currentIdx + 1] = currentRowStart + 1;
					fv[currentIdx + 2] = currentRowStart + 2;
					fv[currentIdx + 3] = nextRowStart + 1;
					fv[currentIdx + 4] = nextRowStart;
					//printf("decrease %i %i %i %i %i\n", fv[currentIdx], fv[currentIdx + 1], fv[currentIdx + 2], fv[currentIdx + 3], fv[currentIdx + 4]);
					
					currentIdx += 5;
					i++;
				}
				else {
					fv[currentIdx] = currentRowStart;
					fv[currentIdx + 1] = currentRowStart + 1;
					fv[currentIdx + 2] = nextRowStart + 1;
					fv[currentIdx + 3] = nextRowStart;
					//printf("quad %i %i %i %i\n", fv[currentIdx], fv[currentIdx + 1], fv[currentIdx + 2], fv[currentIdx + 3]);
					
					currentIdx += 4;
				}
			}
			else {
				fv[currentIdx] = currentRowStart + i;
				fv[currentIdx + 1] = currentRowStart + i + 1;
				fv[currentIdx + 2] = nextRowStart + i + 1 + rowIncrease;
				fv[currentIdx + 3] = nextRowStart + i + rowIncrease;
				//printf("quad %i %i %i %i\n", fv[currentIdx], fv[currentIdx + 1], fv[currentIdx + 2], fv[currentIdx + 3]);
					
				currentIdx += 4;
			}
		}
		rowEnd = nextRowEnd;
		currentRowStart = nextRowStart;
		nextRowStart += rowEnd;	
		
	}
}

short YarnPatch::rowDifference(short & step, short & rowEnd, const short & targetRowEnd, const short & irow) const
{
	if(irow == 0) return 0;
	if(step == 0) return 0;
		
	short rowIncrease = step;
	rowEnd += step;
	if(rowEnd == targetRowEnd) step = 0;
	
	return rowIncrease;
}

void YarnPatch::getCornerUV(short quadV, float & u, float & v) const
{
	if(quadV == 0) {
		u = 0.f;
		v = 0.f;
	}
	else if(quadV == 1) {
		u = 1.f;
		v = 0.f;
	}
	else if(quadV == 2) {
		u = 1.f;
		v = 1.f;
	}
	else {
		u = 0.f;
		v = 1.f;
	}
}

void YarnPatch::increaseWaleGrid(int dv)
{
    short oldv = m_numWaleGrid;
	m_numWaleGrid += dv;
	if(!verifyNumGrid()) m_numWaleGrid = oldv;
}

void YarnPatch::increaseCourseGrid(unsigned v0, unsigned v1, int dv)
{
	if(!hasWaleEdges()) return;
	short oldn[2];
	oldn[0] = m_numCourseGrid[0];
	oldn[1] = m_numCourseGrid[1];
	
	short nw = 0;
	unsigned v[4];
	waleEdges(nw, v);
	if(v[0] == v0 && v[2] == v1) {
		m_numCourseGrid[0] += dv;
	}
	else if(v[0] == v1 && v[2] == v0) {
		m_numCourseGrid[0] += dv;
	}
	else if(v[1] == v0 && v[3] == v1) {
		m_numCourseGrid[1] += dv;
	}
	else if(v[1] == v1 && v[3] == v0) {
		m_numCourseGrid[1] += dv;
	}
	
	if(!verifyNumGrid()) {
		m_numCourseGrid[0] = oldn[0];
		m_numCourseGrid[1] = oldn[1];
	}
}

char YarnPatch::isWaleEdge(unsigned v0, unsigned v1) const
{
	if(!hasWaleEdges()) return 0;
	short nw = 0;
	unsigned v[4];
	waleEdges(nw, v);
	if(v0 == v[0] && v1 == v[1]) return 1;
	if(v0 == v[1] && v1 == v[0]) return 1;
	if(v0 == v[2] && v1 == v[3]) return 1;
	if(v0 == v[3] && v1 == v[2]) return 1;
	return 0;
}

char YarnPatch::isCourseEdge(unsigned v0, unsigned v1) const
{
	if(!hasWaleEdges()) return 0;
	short nw = 0;
	unsigned v[4];
	waleEdges(nw, v);
	if(v0 == v[0] && v1 == v[2]) return 1;
	if(v0 == v[2] && v1 == v[0]) return 1;
	if(v0 == v[1] && v1 == v[3]) return 1;
	if(v0 == v[3] && v1 == v[1]) return 1;
	return 0;
}

short YarnPatch::getWaleGrid() const
{
	return m_numWaleGrid;
}

void YarnPatch::setWaleGrid(short val)
{
	m_numWaleGrid = val;
	if(m_numWaleGrid < 2) m_numWaleGrid = 2;
}

short YarnPatch::getCourseGrid(unsigned v0, unsigned v1) const
{
	if(!hasWaleEdges()) return -1;
	short side = courseSide(v0, v1);
	return m_numCourseGrid[side];
}

char YarnPatch::setCourseGrid(unsigned v0, unsigned v1, short val)
{
	if(!hasWaleEdges()) return 0;
	short side = courseSide(v0, v1);
	short oldn = m_numCourseGrid[side];
	m_numCourseGrid[side] = val;
	if(!verifyNumGrid()) {
		m_numCourseGrid[side] = oldn;
		return 0;
	}
	return 1;
}

short YarnPatch::courseSide(unsigned v0, unsigned v1) const
{
	short nw = 0;
	unsigned v[4];
	waleEdges(nw, v);
	if(v0 == v[0] && v1 == v[2]) return 0;
	if(v0 == v[2] && v1 == v[0]) return 0;
	return 1;
}

char YarnPatch::isTriangle() const
{
    short i, j;
	for(i = 0; i < 4; i++) {
	    j = (i + 1)%4;
	    if(m_quadVertices[i] == m_quadVertices[j]) return 1;
	}
	return 0;
}

void YarnPatch::tessellateQuad()
{
    const short nrow = m_numWaleGrid + 1;
	const short colChange = m_numCourseGrid[1] - m_numCourseGrid[0];
	short colDir = 0;
	if(colChange > 0) colDir = 1;
	else if(colChange < 0) colDir = -1;
	
	const short ncol0 = m_numCourseGrid[0] + 1;
	const short ncol1 = m_numCourseGrid[1] + 1;
	
	fillP(nrow, ncol0, ncol1, colDir);
	fillF(nrow, ncol0, ncol1, colDir);
	
	m_hasTessellation = 1;
}

void YarnPatch::tessellateTriangle()
{
    if(isConverging()) {
        printf("triangle is converging");
    }
    else {
        printf("triangle is diverging");
    }
}

char YarnPatch::isConverging() const
{
    return (m_quadVertices[m_waleVertices[1]] == m_quadVertices[m_waleVertices[3]]);
}
//:~
