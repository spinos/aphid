/*
 *  DrawKdTree.h
 *  
 *
 *  Created by jian zhang on 1/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_KD_TREE_H
#define APH_OGL_DRAW_KD_TREE_H

#include <kd/KdNTree.h>
#include <ogl/DrawBox.h>

namespace aphid {

template<typename T, typename Tn>
class DrawKdTree : public DrawBox {

	KdNTree<T, Tn > * m_tree;
	
public:
	DrawKdTree(KdNTree<T, Tn > * tree);
	
	void drawCells();

private:
	void drawCell(int branchIdx,
					int nodeIdx,
					const BoundingBox & b);
	
};

template<typename T, typename Tn>
DrawKdTree<T, Tn>::DrawKdTree(KdNTree<T, Tn > * tree)
{ 
	m_tree = tree; 
}

template<typename T, typename Tn>
void DrawKdTree<T, Tn>::drawCells()
{
	const BoundingBox & b = m_tree->getBBox();
	
	KdTreeNode * r = m_tree->root()->node(0);
	if(r->isLeaf() ) {
		drawBoundingBox(&b);
		return;
	}
	
	int branchIdx = m_tree->root()->internalOffset(0);
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox leftBox, rightBox;
	b.split(axis, splitPos, leftBox, rightBox);
	
	drawCell(branchIdx, 0, leftBox);
	drawCell(branchIdx, 1, rightBox);
	
}

template<typename T, typename Tn>
void DrawKdTree<T, Tn>::drawCell(int branchIdx,
					int nodeIdx,
					const BoundingBox & b)
{
	Tn * currentBranch = m_tree->branches()[branchIdx];
	KdTreeNode * r = currentBranch->node(nodeIdx);
	if(r->isLeaf() ) {
		drawBoundingBox(&b);
		return;
	}
	
	const int & axis = r->getAxis();
	const float & splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	
	const int & offset = r->getOffset();
	
	if(offset < Tn::TreeletOffsetMask) {
		drawCell(branchIdx,
				nodeIdx + offset,
				lftBox);
		drawCell(branchIdx, 
				nodeIdx + offset + 1, 
				rgtBox);
							
	} else {
		drawCell(branchIdx + offset & Tn::TreeletOffsetMaskTau,
				0,
				lftBox);
							
		drawCell(branchIdx + offset & Tn::TreeletOffsetMaskTau,
				1,
				rgtBox);
				
	}
	
}

}
#endif