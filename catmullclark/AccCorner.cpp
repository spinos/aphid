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

namespace aphid {

AccCorner::AccCorner() {}
AccCorner::~AccCorner() { reset(); }

void AccCorner::setCenterIndex(const int & x)
{
	_centerIndex = x;
}

void AccCorner::setCenterPosition(Vector3F * p)
{
	_centerPosition = p;
}
	
void AccCorner::setCenterNormal(Vector3F * p)
{
	_centerNormal = p;
}

void AccCorner::reset()
{
	_edgePositions.clear();
	_cornerPositions.clear();
	_edgeNormals.clear();
	_cornerNormals.clear();
	_edgeIndices.clear();
	_cornerIndices.clear();
	_tagCornerIndices.clear();
}

void AccCorner::addEdgeNeighbor(int idx, Vector3F * positions, Vector3F * normals)
{    
    _edgeIndices.push_back(idx);
    _edgePositions.push_back(positions[idx]);
	_edgeNormals.push_back(normals[idx]);
}

void AccCorner::addCornerNeighbor(int idx, Vector3F * positions, Vector3F * normals)
{
    for(unsigned i=0; i < _cornerIndices.size(); i++) {
        if(_cornerIndices[i] == idx) return;
    }
    _cornerIndices.push_back(idx);
	_tagCornerIndices.push_back(1);
    _cornerPositions.push_back(positions[idx]);
	_cornerNormals.push_back(normals[idx]);
}

void AccCorner::addCornerNeighborBetween(int a, int b, Vector3F * positions, Vector3F * normals)
{
	_cornerIndices.push_back(a);
	_cornerIndices.push_back(b);
	_tagCornerIndices.push_back(0);
	_tagCornerIndices.push_back(0);
	_cornerPositions.push_back(positions[a] * 0.5f + positions[b] * 0.5f);
	Vector3F an = normals[a] * 0.5f + normals[b] * 0.5f;
	an.normalize();
	_cornerNormals.push_back(an);
}

char AccCorner::isOnBoundary() const
{
    return _cornerPositions.size() != _edgePositions.size();
}

int AccCorner::valence() const
{
	return _edgePositions.size();
}

Vector3F AccCorner::computeNormal() const
{
	if(isOnBoundary())
		return *_centerNormal * (2.f / 3.f) +  _edgeNormals[0] * (1.f / 6.f) +  _edgeNormals[valence() - 1] * (1.f / 6.f);
		
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
	res += *_centerNormal * valence() * valence();
	return res / sum;
}

Vector3F AccCorner::computePosition() const
{
	if(isOnBoundary())
		return *_centerPosition * (2.f / 3.f) +  _edgePositions[0] * (1.f / 6.f) +  _edgePositions[valence() - 1] * (1.f / 6.f);
		
	float e = 4.f;
	float c = 1.f;
	float sum = 0.f;
	Vector3F res;
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
	res += *_centerPosition * valence() * valence();
	res /= sum;
	return res;
}

void AccCorner::edgeNeighborBeside(int nei, int & a, int &b) const
{
	int found = -1;
	for(unsigned i=0; i < _edgeIndices.size(); i++) {
        if(_edgeIndices[i] == nei) {
			found = i;
            break;
		}
    }
	
	if(found < 0) {
		std::clog<<"\nWARNING: AccCorner cannot find neighbor of "<<nei<<" besides "<<a<<"\n";
		return;
	}
	
	int pre = found - 1;
	if(pre < 0) pre = _edgeIndices.size() - 1;
	unsigned post = found + 1;
	if(post > _edgeIndices.size() - 1) post = 0;
	
	a = _edgeIndices[pre];
	b = _edgeIndices[post];
}

const int & AccCorner::edgeIndex(const unsigned & i) const
{
	return _edgeIndices[i];
}

const std::vector<int> & AccCorner::edgeIndices() const { return _edgeIndices; }
const std::vector<int> & AccCorner::cornerIndices() const { return _cornerIndices; }
const std::vector<char> & AccCorner::tagCornerIndices() const { return _tagCornerIndices; }

void AccCorner::verbose() const
{
	std::cout<<"\naround v["<<_centerIndex<<"] n edges "<<_edgePositions.size()<<" n corners "<<_cornerPositions.size();
	std::cout<<"\nedge nei idx: ";
	for(unsigned i=0; i < _edgeIndices.size(); i++) std::cout<<" "<<_edgeIndices[i];
	
	std::cout<<"\ncorner nei idx: ";
	for(unsigned i=0; i < _cornerIndices.size(); i++) std::cout<<" "<<_cornerIndices[i];
    printf("\n");
}

}
