/*
 *  ADistanceField.h
 *  
 *	distance field with function
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "AGraph.h"
#include "BDistanceFunction.h"
#include <Array.h>
#include <Calculus.h>
#include <map>

namespace aphid {

namespace sdf {
enum NodeState {
	StBackGround = 0,
	StFront = 1,
	StUnknown = 2,
	StKnown = 3,
	StFar = 4,
	StVisited = 5
};

}

struct DistanceNode {
	
	Vector3F pos;
	float val;
	short label;
	short stat;
};

struct IDistanceEdge {

	sdb::Coord2 vi; /// ind to node
	float len; /// length of edge
	float err; /// max error of distance always positive
	float cx; /// x [0,1] where front cross
};
 
class ADistanceField : public AGraph<DistanceNode, IDistanceEdge > {

	sdb::Sequence<sdb::Coord2 > m_dirtyEdges;
	float m_mxErr, m_mnErr;
	
public:
	ADistanceField();
	virtual ~ADistanceField();
	
	void nodeColor(Vector3F & dst, const DistanceNode & n,
					const float & scale) const;
	
/// march through unblocked edges
/// negative nodes not visited 
	void markInsideOutside(const int & originNodeInd = -1);
	
	sdb::Sequence<sdb::Coord2 > * dirtyEdges();

/// per dirty edge, at linear center delta x, compare actual distance to recovered distance
	template<typename Tf>
	float estimateError(Tf * func, const float & shellThickness)
	{
		m_dirtyEdges.begin();
		float act, rec, err;
		int c = 0;
		m_mxErr = 0.f;
		m_mnErr = 1.f;
		Vector3F q;
		while(!m_dirtyEdges.end() ) {
			
			const sdb::Coord2 i = m_dirtyEdges.key();
			const DistanceNode & n1 = nodes()[i.x];
			const DistanceNode & n2 = nodes()[i.y];
			
/// front nodes
			if(n1.label == sdf::StFront && n2.label == sdf::StFront) {
/// do not test inside edge
			if(n1.val >= 0.f || n2.val >= 0.f ) {
				Vector3F q = (n1.pos + n2.pos ) * .5f;
								
				act = func->calculateDistance(q) - shellThickness;
				rec = (n1.val + n2.val ) * .5f;
				err = Absolute<float>(rec - act);
				if(m_mxErr < err)
					m_mxErr = err;
				if(m_mnErr > err)
					m_mnErr = err;
					
				int k = edgeIndex(i.x, i.y);
				if(k>-1)
					edges()[k].err = act;
					
				c++;
			}
			}
			
			m_dirtyEdges.next();
		}
		
		std::cout<<"\n n error sample "<<c;
		return m_mxErr;
	}
	
	const float & maxError() const;
	const float & minError() const;
	
	float reconstructError_(const IDistanceEdge * edge) const;
/// x > 0 positive -x 
	void shrinkFront(const float & x);
	
protected:
/// set all background, differentiate known node by distance
	void markUnknownNodes();
	
	template<typename Tf>
	void messureFrontNodes(Tf * func, const float & shellThickness)
	{
		int c = 0;
		const int n = numNodes();
		int i = 0;
		for(;i<n;++i) {
			DistanceNode * d = &nodes()[i];
			if(d->label == sdf::StFront
				&& d->stat == sdf::StUnknown ) {
				d->val = func->calculateDistance(d->pos) - shellThickness;
				d->stat = sdf::StKnown; /// accept
				c++;
			}
		}
		std::cout<<"\n n front sample "<<c;
		
	}
	
///         |						
/// a-------x---->b
///         |
///                  0.5
/// find x if a->b cross front
/// for each edge has both node marked on front
/// find intersect point cx
/// if edge intersects front, march will be blocked	
	template<typename Tf>
	void findFrontEdgeCross(Tf * func, const float & shellThickness)
	{
		int c = 0;
		float beamr;
		const DistanceNode * vs = nodes();
		IDistanceEdge * es = edges();
		m_dirtyEdges.begin();
		while(!m_dirtyEdges.end() ) {
			
			const sdb::Coord2 i = m_dirtyEdges.key();
			const DistanceNode & v1 = vs[i.x];
			const DistanceNode & v2 = vs[i.y];
			IDistanceEdge & e = es[edgeIndex(i.x, i.y)];
			
/// both on front
			if(v1.label == sdf::StFront 
					&& v2.label == sdf::StFront) {	
					
				if(Absolute<float>(v1.val) < 1e-2f)
					e.cx = 1e-3f;
				if(Absolute<float>(v2.val) < 1e-2f)
					e.cx = .999f;
				if(e.cx < 0.f) {
					beamr = e.len * .5f;
					e.cx = func->calculateIntersection(v1.pos, v2.pos,
										beamr, beamr);
				}
				
				if(e.cx >= 0.f)
					c++;
			}
			
			m_dirtyEdges.next();
		}
		std::cout<<"\n n edge cross front "<<c;
		
	}

/// ->	
/// a-----.5---x--b
/// march from positive to negative
/// if mid after x, change sign
/// if x is mid, same sign of (a+b)/2
	void setFrontEdgeSign();
	
/// for each front edge
/// measure distance at .5 .25 .75
/// test against interpolated distance front va,vb 
/// e.err <- max abs(diff)
	template<typename Tf>
	float findEdgeMaxError(Tf * func, const float & shellThickness,
							const Vector3F & pa, const Vector3F & pb,
							const float & a, const float & b)
	{
		float mxErr = 0.f, err, act, rec, alpha;
		for(int i=0; i<3;++i) {
			alpha = calc::UniformlySpacingRecursive16Nodes[i];
			act = func->calculateDistance(pa + pb * alpha) - shellThickness;
			rec = a + b * alpha;
			err = Absolute<float>(act - Absolute<float>(rec) );
			if(mxErr < err)
				mxErr = err;
		}
		return mxErr;
	}
	
	template<typename Tf>
	void estimateFrontEdgeError(Tf * func, const float & shellThickness)
	{
		int c = 0;
		const DistanceNode * vs = nodes();
		IDistanceEdge * es = edges();
		m_dirtyEdges.begin();
		while(!m_dirtyEdges.end() ) {
			
			const sdb::Coord2 i = m_dirtyEdges.key();
			const DistanceNode & v1 = vs[i.x];
			const DistanceNode & v2 = vs[i.y];
			IDistanceEdge & e = es[edgeIndex(i.x, i.y)];
			
/// both on front
			if(v1.label == sdf::StFront 
					&& v2.label == sdf::StFront) {	

/// skip inside					
				if(v1.val <= 0.f && v2.val <= 0.f) {
					e.err = 0.f;
				}
				else {
					e.err = findEdgeMaxError(func, shellThickness, 
												v1.pos, v2.pos - v1.pos, 
												v1.val, v2.val - v1.val);
					c++;
				}
			}
			
			m_dirtyEdges.next();
		}
		std::cout<<"\n n edge error estimate "<<c;
	}
	
/// edges marked to estimate error
	void clearDirtyEdges();
	void addDirtyEdge(const int & a, const int & b);
	
/// propagate distance value	
	void fastMarchingMethod();
	
	bool isNodeBackground(const int & i) const;
	
/// into field
	template<typename Tg, typename Ts>
	void extractGridNodes(DistanceNode * dst, Tg * grd) {
		grd->begin();
		while(!grd->end() ) {
			
			extractGridNodesIn<Ts>(dst, grd->value() );
			
			grd->next();
		}
	}
	
/// out-of field
	template<typename Tg, typename Ts>
	void obtainGridNodeVal(const DistanceNode * src, Tg * grd) {
		grd->begin();
		while(!grd->end() ) {
			
			obtainGridNodeValIn<Ts>(src, grd->value() );
			
			grd->next();
		}
	}

	void snapEdgeToFront();
	int countFrontEdges() const;
	void updateMinMaxError();
	void printErrEdges(const float & thre);
/// pnt where edge cross edge
	Vector3F edgeFrontPos(const IDistanceEdge * e,
							int va, int vb,
							const Vector3F & pa, const Vector3F & pb) const;
					
private:
	void propagate(std::map<int, int > & heap, const int & i);
	void propagateVisit(std::map<int, int > & heap, const int & i);
	int lastBackgroundNode() const;
	void setFarNodeInside();
	void setNodeFar();
	
	template<typename Ts>
	void extractGridNodesIn(DistanceNode * dst,
						sdb::Array<int, Ts> * cell) {
		cell->begin();
		while(!cell->end() ) {
			
			Ts * n = cell->value();
			if(n->index > -1) {
				DistanceNode * d = &dst[n->index];
				d->pos = n->pos;
				d->val = n->val;
			}
			
			cell->next();
		}
	}
		
	template<typename Ts>
	void obtainGridNodeValIn(const DistanceNode * src,
							sdb::Array<int, Ts> * cell) {
		cell->begin();
		while(!cell->end() ) {
			
			Ts * n = cell->value();
			
			n->val = src[n->index].val;
			
			cell->next();
		}
	}
	
/// + -> 0     -
/// - -> 0     +	
/// move node to front to prevent very close cut
	bool snapNodeToFront(DistanceNode & v1, DistanceNode & v2,
						const IDistanceEdge & e);

	void printEdge(const IDistanceEdge * e) const;
	
};

}