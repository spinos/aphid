/*
 *  NTreeDomain.h
 *  
 *
 *  Created by jian zhang on 7/27/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Boundary.h>
#include <KdEngine.h>
#include <ConvexShape.h>

namespace aphid {

template<typename T, typename Tn>
class NTreeDomain : public Domain {

	KdNTree<T, Tn > * m_tree;
	KdEngine m_engine;
	
public:
	NTreeDomain(KdNTree<T, Tn > * tree);
	
	virtual FunctionType functionType() const;
	virtual bool broadphaseIntersect(const BoundingBox & bx);
	virtual float distanceTo(const Vector3F & pref);
	
};

template<typename T, typename Tn>
NTreeDomain<T, Tn >::NTreeDomain(KdNTree<T, Tn > * tree)
{
	m_tree = tree;
	setBBox(tree->getBBox() );
}

template<typename T, typename Tn>
Domain::FunctionType NTreeDomain<T, Tn >::functionType() const
{ return Domain::fnKdTree; }

template<typename T, typename Tn>
bool NTreeDomain<T, Tn >::broadphaseIntersect(const BoundingBox & bx)
{
	if(!getBBox().intersect(bx) )
		return false;
	
	return m_engine.broadphase(m_tree, bx);
}

template<typename T, typename Tn>
float NTreeDomain<T, Tn >::distanceTo(const Vector3F & pref)
{ return -1.f; } //return m_tree->distanceTo(pref); }

}
