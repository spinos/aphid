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
#include <geom/SelectionContext.h>

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
	
	Vector3F aggregatedNormal();
	bool normalDistributeBelow(const float threshold);
 
protected:
	const TreeTyp * tree() const;
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
	m_selectCtx.reset(bx, SelectionContext::Append);
	
	KdEngine::broadphaseSelect(m_tree, &m_selectCtx);
	return numSelected() > 0;
}

template<typename T, typename Tn>
unsigned SelectEngine<T, Tn>::numSelected()
{ return m_selectCtx.numSelected(); }

template<typename T, typename Tn>
const KdNTree<T, Tn > * SelectEngine<T, Tn>::tree() const
{ return m_tree; }

template<typename T, typename Tn>
KdNTree<T, Tn > * SelectEngine<T, Tn>::tree()
{ return m_tree; }

template<typename T, typename Tn>
const sdb::VectorArray<T> & SelectEngine<T, Tn>::source() const
{ return *m_tree->source(); }

template<typename T, typename Tn>
sdb::Sequence<int> * SelectEngine<T, Tn>::primIndices()
{ return m_selectCtx.primIndices(); }

template<typename T, typename Tn>
Vector3F SelectEngine<T, Tn>::aggregatedNormal()
{
	sdb::Sequence<int> * prims = primIndices();
	const sdb::VectorArray<T> & src = source();
	
	Vector3F agn(0.f, 0.f, 0.f);

	prims->begin();
	while(!prims->end() ) {
	
		const T * ts = src[prims->key()];
		agn += ts->calculateNormal();
	
		prims->next();
	}
	
	agn.normalize();
	return agn;
}

template<typename T, typename Tn>
bool SelectEngine<T, Tn>::normalDistributeBelow(const float threshold)
{
	const Vector3F agn = aggregatedNormal();
	sdb::Sequence<int> * prims = primIndices();
	const sdb::VectorArray<T> & src = source();
	
	prims->begin();
	while(!prims->end() ) {
	
		const T * ts = src[prims->key()];
		float ndn = ts->calculateNormal().dot(agn);
	
		if(ndn < threshold) {
			return true;
		}
		
		prims->next();
	}
	
	return false;
}

}
#endif
