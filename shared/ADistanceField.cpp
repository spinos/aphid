/*
 *  ADistanceField.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ADistanceField.h"

namespace aphid {

ADistanceField::ADistanceField()
{}

ADistanceField::~ADistanceField()
{ clearDirtyEdges(); }

sdb::Array<sdb::Coord2, EdgeRec > * ADistanceField::dirtyEdges()
{ return &m_dirtyEdges; }

void ADistanceField::nodeColor(Vector3F & dst, const DistanceNode & n,
						const float & scale) const
{	
	if(n.stat == sdf::StUnknown)
		return dst.set(.3f, .3f, .3f);
	
	if(n.val > 0.f) {
		float r = MixClamp01F<float>(1.f, 0.f, n.val * scale);
		dst.set(r, 0.f, 0.f );
	}
	else {
		float b = MixClamp01F<float>(1.f, 0.f, -n.val * scale);
		dst.set(0.f, .5f * (1.f - b), .5f + b * .5f);
	}
}

void ADistanceField::initNodes()
{
	const int n = numNodes();
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		d.val = 1e9f;
		d.label = sdf::StBackGround;
		d.stat = sdf::StUnknown;
	}
}

void ADistanceField::fastMarchingMethod()
{
/// heap of trial
	std::map<int, int> trials;
	const int n = numNodes();
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		if(d.stat == sdf::StKnown) {
			propagate(trials, i);
		}
	}
	
/// for each trial
	while (trials.size() > 0) {

/// A is first in trial		
		i = trials.begin()->first;
/// distance is known after propagation
		nodes()[i].stat = sdf::StKnown;
/// remove A from trial
		trials.erase(trials.begin() );
		
/// from A
		propagate(trials, i);
		
		//std::cout<<"\n trial n "<<trials.size();
		//std::cout.flush();
	}
}

/// A to B
void ADistanceField::propagate(std::map<int, int > & heap, 
												const int & i)
{
	const DistanceNode & A = nodes()[i];
	
/// for each neighbor of A
	const int endj = edgeBegins()[i+1];
	int vj, j = edgeBegins()[i];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == i)
			vj = eg.vi.y;
			
		DistanceNode & B = nodes()[vj];
		if(B.stat == sdf::StUnknown) {
		
/// min distance to B via A
/// need eikonal approximation here
			if(A.val + eg.len < B.val)
				B.val = A.val + eg.len;
				
/// add to trial
			heap[vj] = 0;
		}
	}
}

void ADistanceField::propagateVisit(std::map<int, int > & heap, const int & i)
{
	const DistanceNode & A = nodes()[i];
	
/// for each neighbor of A
	const int endj = edgeBegins()[i+1];
	int vj, j = edgeBegins()[i];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == i)
			vj = eg.vi.y;
			
		DistanceNode & B = nodes()[vj];
	
/// do not visit inside
		if( B.val >= 0.f
			&& B.stat == sdf::StFar) 
			heap[vj] = 0;
	}
}

/// Dijkstra
void ADistanceField::markInsideOutside(const int & originNodeInd)
{
	setNodeFar();
	int i = originNodeInd;
	if(i < 0) {
		i = lastBackgroundNode();
	
		DistanceNode & ln = nodes()[i];
		std::cout<<"\n progress from "<<ln.pos;
		std::cout.flush();
	}
	
/// heap of trial
	std::map<int, int> trials;
	trials[i] = 0;
	
/// for each trial
	while (trials.size() > 0) {

/// A is first in trial		
		i = trials.begin()->first;

		nodes()[i].stat = sdf::StVisited;
/// remove A from trial
		trials.erase(trials.begin() );
		
/// from A
		propagateVisit(trials, i);
		
		//std::cout<<"\n trial n "<<trials.size();
		//std::cout.flush();
	}
	
/// negate not visited
	setFarNodeInside();
	
}

/// un-visit all
void ADistanceField::setNodeFar()
{
	const int n = numNodes();
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		d.stat = sdf::StFar;
	}
}

int ADistanceField::lastBackgroundNode() const
{
	int i = numNodes() - 1;
	for(;i>0;--i) {
		if(nodes()[i].label == sdf::StBackGround)	
			return i;
	}
	return 0;
}

void ADistanceField::setFarNodeInside()
{
	const int n = numNodes();
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		if(d.stat == sdf::StFar) {
/// inside distance is negative
			if(d.val > 0.f)
				d.val = -d.val;
		}
	}
}

void ADistanceField::clearDirtyEdges()
{ m_dirtyEdges.clear(); }

void ADistanceField::addDirtyEdge(const int & a, const int & b)
{ 
	sdb::Coord2 k = sdb::Coord2(a, b).ordered();
	EdgeRec * e = m_dirtyEdges.find(k);
	if(!e) {
		e = new EdgeRec;
		e->key = k;
		m_dirtyEdges.insert(k, e );
	}
}	  

const float & ADistanceField::maxError() const
{ return m_mxErr; }

const float & ADistanceField::minError() const
{ return m_mnErr; }

bool ADistanceField::isNodeBackground(const int & i) const
{ return nodes()[i].label == sdf::StBackGround; }

bool ADistanceField::snapNodeToFront(DistanceNode & v1, DistanceNode & v2,
									const IDistanceEdge & e)
{	
	if(Absolute<float>(v1.val) + Absolute<float>(v2.val) < e.len * .7f)
		return false;
	
	const Vector3F dv = v2.pos - v1.pos;
	const float eps = e.len * .2f;
	
	if(v1.val > 0.f) {
		if(v1.val < eps) {
		
			v1.pos += dv.normal() * v1.val;
			v1.val = 0.f;
			return true;
		}
		else if(-v2.val < eps) {
		
			v2.pos += dv.normal() * v2.val;
			v2.val = 0.f;
			return true;
		}
	}
	else {
		if(v2.val < eps) {
		
			v2.pos -= dv.normal() * v2.val;
			v2.val = 0.f;
			return true;
		}
		else if(-v1.val < eps) {
		
			v1.pos -= dv.normal() * v1.val;
			v1.val = 0.f;
			return true;
		}
	}
	
	return false;
}

int ADistanceField::countFrontEdges()
{
	int c = 0;
	DistanceNode * v = nodes();
	IDistanceEdge * es = edges();
	const int ne = numEdges();
	for(int i=0; i< ne;++i) {
		
		IDistanceEdge & e = es[i];
		
		DistanceNode & v1 = v[e.vi.x];
		DistanceNode & v2 = v[e.vi.y];
		
/// cross front
		if(v1.val * v2.val < 0.f) {			
			if(!snapNodeToFront(v1, v2, e) )
				c++;			
		}
	}
	
	return c;
}

}