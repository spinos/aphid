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

namespace aphid {

template<typename T>
class KdEngine {

public:
	KdEngine();
	virtual ~KdEngine();
	
	void buildTree(KdNTree<T, KdNode4 > * tree, 
					sdb::VectorArray<T> * source, const BoundingBox & box,
					int maxLeafPrim = 8,
					int maxLevel = 8);
	
	void printTree(KdNTree<T, KdNode4 > * tree);
	// typedef KdNTree<T, KdNode4 > TreeType;
    
protected:

private:
	void printBranch(KdNTree<T, KdNode4 > * tree, int idx);
	
};

template<typename T>
KdEngine<T>::KdEngine() {}

template<typename T>
KdEngine<T>::~KdEngine() {}

template<typename T>
void KdEngine<T>::buildTree(KdNTree<T, KdNode4 > * tree, 
							sdb::VectorArray<T> * source, const BoundingBox & box,
							int maxLeafPrim, int maxLevel)
{
	tree->init(source, box);
    KdNBuilder<4, T, KdNode4 > bud;
	bud.SetNumPrimsInLeaf(maxLeafPrim);
	bud.MaxTreeletLevel = maxLevel;
	
/// first split
	SahSplit<T> splt(source);
	splt.initIndicesAndBoxes(source->size() );
    splt.setBBox(box);
	if(splt.numPrims() > 1024) splt.compressPrimitives();
	
	SahSplit<T>::GlobalSplitContext = &splt;
	
	bud.build(&splt, tree);
	tree->verbose();
}

template<typename T>
void KdEngine<T>::printTree(KdNTree<T, KdNode4 > * tree)
{
	KdNode4 * tn = tree->root();
	std::cout<<"\n root";
	tn->verbose();
	int i=0;
	KdTreeNode * child = tn->node(0);
	if(child->isLeaf() ) {}
	else {
		printBranch(tree, tn->internalOffset(0) );
	}
}

template<typename T>
void KdEngine<T>::printBranch(KdNTree<T, KdNode4 > * tree, int idx)
{
	KdNode4 * tn = tree->nodes()[idx];
	std::cout<<"\n branch["<<idx<<"]";
	tn->verbose();
	int i=14;
	for(;i<KdNode4::NumNodes;++i) {
		KdTreeNode * child = tn->node(i);
		if(child->isLeaf() ) {}
		else {
			if(tn->internalOffset(i) > 0) printBranch(tree, idx + tn->internalOffset(i) );
		}
	}
}

}
//:~