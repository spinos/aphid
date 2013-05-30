/*
 *  AccCorner.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 5/30/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "AccCorner.h"

void AccCorner::reset()
{
	_numEdgeNei = 0;
    _numCornerNei = 0;
}

void AccCorner::addEdgeNeighbor(int idx, Vector3F * positions, Vector3F * normals)
{
    if(_numEdgeNei >= 5) return;
        
    _edgeIndices[_numEdgeNei] = idx;
    _edgePositions[_numEdgeNei] = positions[idx];
	_edgeNormals[_numEdgeNei] = normals[idx];
    _numEdgeNei++;
}

void AccCorner::addCornerNeighbor(int idx, Vector3F * positions, Vector3F * normals)
{
    if(_numCornerNei >= 5) return;
    
    for(int i=0; i < _numCornerNei; i++) {
        if(_cornerIndices[i] == idx)
            return;
    }
    
    _cornerIndices[_numCornerNei] = idx;
    _cornerPositions[_numCornerNei] = positions[idx];
	_cornerNormals[_numCornerNei] = normals[idx];
    _numCornerNei++;
}

void AccCorner::addCornerNeighborBetween(int a, int b, Vector3F * positions, Vector3F * normals)
{
    if(_numCornerNei >= 5) return;
    _cornerPositions[_numCornerNei] = positions[a] * 0.5f + positions[b] * 0.5f;
	_cornerNormals[_numCornerNei] = normals[a] * 0.5f + normals[b] * 0.5f;
	_cornerNormals[_numCornerNei].normalize();
    _numCornerNei++;
}

char AccCorner::isOnBoundary() const
{
    return _numCornerNei != _numEdgeNei;
}

int AccCorner::valence() const
{
	return _numEdgeNei;
}

Vector3F AccCorner::computeNormal() const
{
	if(isOnBoundary())
		return _centerNormal * (2.f / 3.f) +  _edgeNormals[0] * (1.f / 6.f) +  _edgeNormals[_numEdgeNei - 1] * (1.f / 6.f);
		
	float e = 4.f;
	float c = 1.f;
	float sum = 0.f;
	Vector3F res;
	res.setZero();
	Vector3F q;
	
	for(int i = 0; i < valence(); i++) {
		q = _edgeNormals[i];
		res += q * e;
		sum += e;
		q = _cornerNormals[i];
		res += q * c;
		sum += c;
	}
	
	sum += valence() * valence();
	res += _centerNormal * valence() * valence();
	return res / sum;
}

Vector3F AccCorner::computePosition() const
{
	if(isOnBoundary())
		return _centerPosition * (2.f / 3.f) +  _edgePositions[0] * (1.f / 6.f) +  _edgePositions[_numEdgeNei - 1] * (1.f / 6.f);
		
	float e = 4.f;
	float c = 1.f;
	float sum = 0.f;
	Vector3F res;
	res.setZero();
	Vector3F q;
	
	for(int i = 0; i < valence(); i++) {
		q = _edgePositions[i];
		res += q * e;
		sum += e;
		q = _cornerPositions[i];
		res += q * c;
		sum += c;
	}
	
	sum += valence() * valence();
	res += _centerPosition * valence() * valence();
	return res / sum;
}

void AccCorner::edgeNeighborBeside(int nei, int & a, int &b) const
{
	int found = -1;
	for(int i=0; i < _numEdgeNei; i++) {
        if(_edgeIndices[i] == nei) {
			found = i;
            break;
		}
    }
	
	if(found < 0) return;
	
	int pre = found - 1;
	if(pre < 0) pre = _numEdgeNei - 1;
	int post = found + 1;
	if(post > _numEdgeNei - 1) post = 0;
	
	a = _edgeIndices[pre];
	b = _edgeIndices[post];
}

void AccCorner::verbose() const
{
	printf("\nedge nei: ");
	for(int i=0; i < _numEdgeNei; i++) printf(" %i ", _edgeIndices[i]);
	
	printf("\ncorner nei: ");
	for(int i=0; i < _numCornerNei; i++) printf(" %i ", _cornerIndices[i]);
    printf("\n");
}
