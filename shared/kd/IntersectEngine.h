/*
 *  IntersectEngine.h
 *  
 *
 *  Created by jian zhang on 1/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_KD_INTERSECT_ENGINE_H
#define APH_KD_INTERSECT_ENGINE_H

#include <kd/KdEngine.h>

namespace aphid {

template<typename T, typename Tn>
class IntersectEngine : public KdEngine {

typedef KdNTree<T, Tn > TreeTyp;

	BoxIntersectContext m_boxCtx;
	TreeTyp * m_tree;
	
public:
	IntersectEngine(TreeTyp * tree);
	
	bool intersect(const BoundingBox & box);
    bool intersect(const cvx::Tetrahedron & tet);
	
	const BoundingBox & getBBox() const;

};

template<typename T, typename Tn>
IntersectEngine<T, Tn>::IntersectEngine(TreeTyp * tree)
{
	m_tree = tree;
}

template<typename T, typename Tn>
bool IntersectEngine<T, Tn>::intersect(const BoundingBox & box)
{
	m_boxCtx = box;
	m_boxCtx.reset();
	intersectBox(m_tree, &m_boxCtx);
	return m_boxCtx.numIntersect() > 0;
}

template<typename T, typename Tn>
bool IntersectEngine<T, Tn>::intersect(const cvx::Tetrahedron & tet)
{
    return false;
}

template<typename T, typename Tn>
const BoundingBox & IntersectEngine<T, Tn>::getBBox() const
{ return m_tree->getBBox(); }

}
#endif
