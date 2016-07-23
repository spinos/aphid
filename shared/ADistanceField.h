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

struct EdgeRec {
	sdb::Coord2 key;
	float val;
};
 
class ADistanceField : public AGraph<DistanceNode> {

	sdb::Array<sdb::Coord2, EdgeRec > m_dirtyEdges;
	float m_mxErr, m_mnErr;
	
public:
	ADistanceField();
	virtual ~ADistanceField();
	
	void nodeColor(Vector3F & dst, const DistanceNode & n,
					const float & scale) const;
	
	void markInsideOutside(const int & originNodeInd = -1);
	
	sdb::Array<sdb::Coord2, EdgeRec > * dirtyEdges();

/// per dirty edge, at linear center delta x, compare actual distance to recovered distance
	template<typename Tf>
	float estimateError(const Tf * func, const float & shellThickness,
						const float & h)
	{
		m_dirtyEdges.begin();
		float act, rec, err;
		m_mxErr = 0.f;
		m_mnErr = 10.f;
		Vector3F q;
		while(!m_dirtyEdges.end() ) {
			
			const sdb::Coord2 i = m_dirtyEdges.key();
			const DistanceNode & n1 = nodes()[i.x];
			const DistanceNode & n2 = nodes()[i.y];
			
			err = 0.f;
			
/// front nodes
			if(n1.label == sdf::StFront
				&& n2.label == sdf::StFront) {
/// do not test inside edge
			if(n1.val * n2.val < 0.f
			|| (n1.val > 0.f && n2.val > 0.f)  ) {
				Vector3F q = (n1.pos + n2.pos ) * .5f;
								
				act = func->calculateDistance(q) - shellThickness;
				rec = (n1.val + n2.val ) * .5f;
				err = Absolute<float>(rec - act) / h;
				if(m_mxErr < err)
					m_mxErr = err;
				if(m_mnErr > err)
					m_mnErr = err;
			}
			}
				
			m_dirtyEdges.value()->val = err;
			
			m_dirtyEdges.next();
		}
		return m_mxErr;
	}
	
	const float & maxError() const;
	const float & minError() const;
	
protected:
	void initNodes();
	
	template<typename Tf>
	void calculateDistanceOnFront(const Tf * func, const float & shellThickness)
	{
		const int n = numNodes();
		int i = 0;
		for(;i<n;++i) {
			DistanceNode * d = &nodes()[i];
			if(d->label == sdf::StFront) {
				d->val = func->calculateDistance(d->pos) - shellThickness;
				d->stat = sdf::StKnown; /// accept
			}
		}
	}
	
/// edges marked to estimate error
	void clearDirtyEdges();
	void addDirtyEdge(const int & a, const int & b);
	
/// propagate distance value	
	void fastMarchingMethod();
	
private:
	void propagate(std::map<int, int > & heap, const int & i);
	void propagateVisit(std::map<int, int > & heap, const int & i);
	int lastBackgroundNode() const;
	void setFarNodeInside();
	void setNodeFar();
	
};

}