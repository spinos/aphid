/*
 *  SelectEngine.h
 *  
 *	kd-tree selection
 *
 *  Created by jian zhang on 1/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_KD_SELECT_ENGINE_H
#define APH_KD_SELECT_ENGINE_H

#include <kd/KdEngine.h>
#include <SelectionContext.h>

namespace aphid {

template<typename T, typename Tn>
class SelectEngine : public KdEngine {

public:
typedef KdNTree<T, Tn > TreeTyp;

private:
	SphereSelectionContext m_selectCtx;
	TreeTyp * m_tree;
	
public:
	SelectEngine(TreeTyp * tree);
	
	bool select(const Vector3F & center,
				const float & radius);
				
	bool select(const BoundingBox & bx);
				
	unsigned numSelected();
 
protected:
	TreeTyp * tree();
	const sdb::VectorArray<T> & source() const;
	sdb::Sequence<int> * primIndices();
	
};

template<typename T, typename Tn>
SelectEngine<T, Tn>::SelectEngine(TreeTyp * tree)
{
	m_tree = tree;
}

template<typename T, typename Tn>
bool SelectEngine<T, Tn>::select(const Vector3F & center,
				const float & radius)
{
	m_selectCtx.deselect();
	m_selectCtx.reset(center, radius, SelectionContext::Append);
		
	KdEngine::select(m_tree, &m_selectCtx);
	return numSelected() > 0;
}

template<typename T, typename Tn>
bool SelectEngine<T, Tn>::select(const BoundingBox & bx)
{
	m_selectCtx.deselect();
	m_selectCtx.reset(bx.center(), bx.radius(), SelectionContext::Append);
		
	KdEngine::select(m_tree, &m_selectCtx);
	return numSelected() > 0;
}

template<typename T, typename Tn>
unsigned SelectEngine<T, Tn>::numSelected()
{ return m_selectCtx.numSelected(); }

template<typename T, typename Tn>
KdNTree<T, Tn > * SelectEngine<T, Tn>::tree()
{ return m_tree; }

template<typename T, typename Tn>
const sdb::VectorArray<T> & SelectEngine<T, Tn>::source() const
{ return *m_tree->source(); }

template<typename T, typename Tn>
sdb::Sequence<int> * SelectEngine<T, Tn>::primIndices()
{ return m_selectCtx.primIndices(); }

}
#endif
