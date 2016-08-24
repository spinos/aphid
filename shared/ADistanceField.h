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

	sdb::Coord2 vi;
	float len;
	float val;
	float minVal;
};
 
class ADistanceField : public AGraph<DistanceNode, IDistanceEdge > {

	sdb::Sequence<sdb::Coord2 > m_dirtyEdges;
	float m_mxErr, m_mnErr;
	
public:
	ADistanceField();
	virtual ~ADistanceField();
	
	void nodeColor(Vector3F & dst, const DistanceNode & n,
					const float & scale) const;
	
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
					edges()[k].val = act;
					
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
	
	float reconstructError(const IDistanceEdge * edge) const;
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
	
///          |						
/// a----x---|-----b
///          |
///                  0.5
///       0.25                 0.75
/// 0.125      0.375     0.625      0.875
/// measure distance at 7 x points
/// as minVal of edge if below threshold
	template<typename Tf>
	void measureDistanceToEdge(Tf * func, const float & shellThickness,
								IDistanceEdge * e,
								const Vector3F & ca,
								const Vector3F & cb,
								const float & thre)
	{
		int i;
		for(i=0; i<8;++i) {
			float md = func->calculateDistance(ca + cb * calc::UniformlySpacingRecursive16Nodes[i]) 
										- shellThickness;
			if(i<1) {
/// first is mid point
				e->val = md;
			}
			
			if(md <= thre) {
/// close enough
				e->minVal = md;
				return;
			}
		}
		
	}

/// for each edge has both node marked on front
/// if distance less than |a,b| / 16
/// edge intersects front, march will be blocked	
	template<typename Tf>
	void measureFrontEdges(Tf * func, const float & shellThickness)
	{
		int c = 0;
		Vector3F a, b;
		float thre;
		const DistanceNode * v = nodes();
		IDistanceEdge * es = edges();
		const int ne = numEdges();
		for(int i=0; i< ne;++i) {
			
			IDistanceEdge & e = es[i];
			
			const DistanceNode & v1 = v[e.vi.x];
			const DistanceNode & v2 = v[e.vi.y];
			
/// both on front
			if(v1.label == sdf::StFront 
					&& v2.label == sdf::StFront) {			
				
				a = v1.pos;
				b = v2.pos - a;
				thre = b.length() * .0625f;
				
				e.minVal = v1.val;
					
				if(v2.val <= thre)
					e.minVal = v2.val;
					
				measureDistanceToEdge(func, shellThickness,
										&e, a, b, thre);
				
/// relative to thre
				e.minVal /= thre;
				if(e.minVal <= 1.f)
					c++;
				
			}
		}
		std::cout<<"\n n edge cross front "<<c;
		
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

};

}