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

sdb::Sequence<sdb::Coord2 > * ADistanceField::dirtyEdges()
{ return &m_dirtyEdges; }

void ADistanceField::nodeColor(Vector3F & dst, const DistanceNode & n,
						const float & scale) const
{	
	if(n.stat == sdf::StUnknown)
		return dst.set(.3f, .3f, .3f);
	
	if(n.val > 0.f) {
		float r = MixClamp01F<float>(1.f, 0.f, n.val * scale);
		dst.set(r, 1.f - r, 0.f );
	}
	else {
		float b = MixClamp01F<float>(1.f, 0.f, -n.val * scale);
		dst.set(0.f, .5f * (1.f - b), .5f + b * .5f);
	}
}

void ADistanceField::markUnknownNodes()
{
	const int n = numNodes();
	int ck = 0;
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		d.label = sdf::StBackGround;
/// decide by distance
		d.stat = d.val > 1e8f ? sdf::StUnknown : sdf::StKnown;
		if(d.stat == sdf::StKnown) ck++;
	}
	std::cout<<"\n known node "<<ck<<"/"<<n;
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
/// do not cross front
		if(eg.cx < 0.f) {
/// do not visit inside
			if( B.val > 1e-3f && B.stat == sdf::StFar) 
				heap[vj] = 0;
		}
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
{ m_dirtyEdges.insert(sdb::Coord2(a, b).ordered() ); }	  

const float & ADistanceField::maxError() const
{ return m_mxErr; }

const float & ADistanceField::minError() const
{ return m_mnErr; }

bool ADistanceField::isNodeBackground(const int & i) const
{ return nodes()[i].label == sdf::StBackGround; }

bool ADistanceField::snapNodeToFront(DistanceNode & v1, DistanceNode & v2,
									const IDistanceEdge & e)
{
/// no cross front 
	if(e.cx < 0)
		return false;
	
	const Vector3F dv = v2.pos - v1.pos;
/// angle of crossing
	float glancing = (Absolute<float>(v1.val) + Absolute<float>(v2.val) ) / e.len;
	float eps = .31f * glancing;

	if(e.cx < eps) {
		v1.pos += dv.normal() * (e.len * e.cx);
		v1.val = 0.f;
		return true;
	}
	
	if(e.cx > 1.f - eps) {
		v2.pos += dv.normal() * (e.len * (e.cx - 1.f));
		v2.val = 0.f;
		return true;
	}
	
	return false;
}

void ADistanceField::snapEdgeToFront()
{//return;
	DistanceNode * v = nodes();
	IDistanceEdge * es = edges();
	const int ne = numEdges();
	for(int i=0; i< ne;++i) {
		
		IDistanceEdge & e = es[i];
		
		DistanceNode & v1 = v[e.vi.x];
		DistanceNode & v2 = v[e.vi.y];
		
/// cross front
		if(v1.val * v2.val < 0.f) {			
			snapNodeToFront(v1, v2, e);
		}
	}
}

int ADistanceField::countFrontEdges() const
{
	int c = 0;
	const DistanceNode * v = nodes();
	const IDistanceEdge * es = edges();
	const int ne = numEdges();
	for(int i=0; i< ne;++i) {
		
		const IDistanceEdge & e = es[i];
		
		const DistanceNode & v1 = v[e.vi.x];
		const DistanceNode & v2 = v[e.vi.y];
		
/// cross front
		if(v1.val * v2.val < 0.f) {			
			c++;			
		}
	}
	
	return c;
}

float ADistanceField::reconstructError_(const IDistanceEdge * edge) const
{
/// unknown distance
	if(edge->err > 1e8f)
		return 0.f;
		
	if(edge->cx < 0.f)
		return 0.f;
	
	const DistanceNode & v1 = nodes()[edge->vi.x];
	const DistanceNode & v2 = nodes()[edge->vi.y];
	
/// skip inside		
	if(v1.val < 0.f && v2.val < 0.f)
		return 0.f;

	if(v1.label == sdf::StFront 
					&& v2.label == sdf::StFront)
		
			return Absolute<float>((v1.val + v2.val) * .5f - edge->err);
	
	return 0.f;
}

void ADistanceField::shrinkFront(const float & x)
{
	const int & n = numNodes();
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		if(d.val > x)
			d.val -= x;
	}
}

void ADistanceField::updateMinMaxError()
{
	m_mxErr = 0.f;
	m_mnErr = 1e8f;
		
	m_dirtyEdges.begin();
	while(!m_dirtyEdges.end() ) {
		
		const sdb::Coord2 i = m_dirtyEdges.key();
		const IDistanceEdge * ae = edge(i.x, i.y );
		
		const float & err = ae->err;
			
		if(m_mxErr < err)
			m_mxErr = err;
		if(m_mnErr > err)
			m_mnErr = err;
		
		m_dirtyEdges.next();
	}
}

void ADistanceField::setFrontEdgeSign()
{
std::cout<<"\n ADistanceField::setFrontEdgeSign() begin"<<std::endl;
	int c = 0;
	//float s;
	const DistanceNode * vs = nodes();
	IDistanceEdge * es = edges();
		
	m_dirtyEdges.begin();
	while(!m_dirtyEdges.end() ) {
	
		const sdb::Coord2 i = m_dirtyEdges.key();
		const DistanceNode & v1 = vs[i.x];
		const DistanceNode & v2 = vs[i.y];
		if(v1.label == sdf::StFront 
					&& v2.label == sdf::StFront) {
		const int j = edgeIndex(i.x, i.y);
		
		IDistanceEdge & e = es[j];
			
/// cross front
		
			float o = e.err;
			
			SameSign<float>(e.err, v1.val*.5f + v2.val * .5f);
			
			if(o*e.err < 0) {
				//std::cout<<"\n e"<<reconstructError(&e);
				c++;
			}
			
		}
		
		m_dirtyEdges.next();
	}
	std::cout<<"\n n edge cross "<<c;
std::cout<<"\n ADistanceField::setFrontEdgeSign() end"<<std::endl;	
}

void ADistanceField::printEdge(const IDistanceEdge * e) const
{
	const sdb::Coord2 i = e->vi;
	const DistanceNode & v1 = nodes()[i.x];
	const DistanceNode & v2 = nodes()[i.y];
	std::cout<<"\n "<<v1.val<<" "<<v2.val
		<<" "<<e->cx<<" "<<e->len<<" "<<e->err;
}

void ADistanceField::printErrEdges(const float & thre)
{
	std::cout<<"\n ADistanceField::printErrEdges begin "<<thre
			<<" "<<m_dirtyEdges.size();
	int c = 0;
	const DistanceNode * vs = nodes();
	IDistanceEdge * es = edges();
		
	m_dirtyEdges.begin();
	while(!m_dirtyEdges.end() ) {
		
		const sdb::Coord2 i = m_dirtyEdges.key();
		const IDistanceEdge * ae = edge(i.x, i.y );
		if(ae ) {
			
			if(ae->err > thre ) {
				std::cout<<"\n err "<<ae->err;
				printEdge(ae);
				c++;
			}
		}
		m_dirtyEdges.next();
	}
	std::cout<<"\n n err edge "<<c
	<<"\n ADistanceField::printErrEdges end "<<std::endl;
}

Vector3F ADistanceField::edgeFrontPos(const IDistanceEdge * e,
							int va, int vb,
							const Vector3F & pa, const Vector3F & pb) const
{
	const float & alpha = e->cx;
/// reverse 
	if(e->vi.x == va)
		return pa * (1.f - alpha) + pb * alpha;
		
	return pa * alpha + pb * (1.f - alpha);
}

bool ADistanceField::isNodeInsideFrontBoundary(int vi) const
{
	const int endj = edgeBegins()[vi+1];
	int vj, j = edgeBegins()[vi];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == vi)
			vj = eg.vi.y;
			
		const DistanceNode & B = nodes()[vj];
		if(B.label != sdf::StFront) {
/// connected to background
			return false;
		}
	}
	return true;
}

bool ADistanceField::isNodeOnFrontBoundary(int vi) const
{
/// not on front
	if(nodes()[vi].label != sdf::StFront)
		return false;
		
	const int endj = edgeBegins()[vi+1];
	int vj, j = edgeBegins()[vi];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == vi)
			vj = eg.vi.y;
			
		const DistanceNode & B = nodes()[vj];
		if(B.label != sdf::StFront) {
/// connected to background
			return true;
		}
	}
	return false;
}

const IDistanceEdge * ADistanceField::longestConnectedEdge(int vi) const
{
	const IDistanceEdge * r = NULL;
	const int endj = edgeBegins()[vi+1];
	int vj, j = edgeBegins()[vi];
	for(;j<endj;++j) {
		
		int k = edgeIndices()[j];

		const IDistanceEdge & eg = edges()[k];
		
		vj = eg.vi.x;
		if(vj == vi)
			vj = eg.vi.y;
			
		const DistanceNode & B = nodes()[vj];
		if(!r) r = & eg;
		else {
			if(eg.len < r->len)
				r = & eg;
		} 
		
	}
	
	return r;
}

}