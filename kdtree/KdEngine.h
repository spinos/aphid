/*
 *  KdEngine.h
 *  testntree
 *
 *  Created by jian zhang on 11/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include "KdNTree.h"
#include "KdBuilder.h"
#include "KdScreen.h"

template<typename T>
class KdEngine {

	KdNTree<T, KdNode4 > * m_tree;
	KdScreen<KdNTree<T, KdNode4 > > * m_screen;
    
public:
	KdEngine();
	virtual ~KdEngine();
	
	void initGeometry(VectorArray<T> * source, const BoundingBox & box);
	void initScreen(int w, int h);
	void render(const Frustum & f);
	KdNTree<T, KdNode4 > * tree();
	
protected:

private:

};

template<typename T>
KdEngine<T>::KdEngine()
{
	m_tree = new KdNTree<T, KdNode4 >();
    m_screen = new KdScreen<KdNTree<T, KdNode4 > >();
}

template<typename T>
KdEngine<T>::~KdEngine()
{
	delete m_tree;
    delete m_screen;
}

template<typename T>
void KdEngine<T>::initGeometry(VectorArray<T> * source, const BoundingBox & box)
{
	m_tree->init(source, box);
    KdNBuilder<4, T, KdNode4 > bud;
	bud.SetNumPrimsInLeaf(8);
	
	SahSplit<T> splt(source->size(), source);
	splt.initIndices();
    splt.setBBox(box);
	bud.build(&splt, m_tree);
	m_tree->verbose();
}

template<typename T>
void KdEngine<T>::initScreen(int w, int h)
{ m_screen->create(w, h); }

template<typename T>
KdNTree<T, KdNode4 > * KdEngine<T>::tree()
{ return m_tree; }

template<typename T>
void KdEngine<T>::render(const Frustum & f)
{
    m_screen->setView(f);
    m_screen->getVisibleFrames(m_tree);
}
//:~