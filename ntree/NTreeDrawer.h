/*
 *  NTreeDrawer.h
 *  testntree
 *
 *  Created by jian zhang on 3/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <KdNTree.h>
#include <DrawBox.h>

namespace aphid {

class NTreeDrawer : public DrawBox {

public:
	NTreeDrawer() {}
	virtual ~NTreeDrawer() {}
	
	template<typename T>
	void drawTree(KdNTree<T, KdNode4 > * tree,
					const Vector3F & origin = Vector3F(0.f, 0.f, 0.f),
					const float & scaling = 1.f);
	
private:
	template<typename T>
	void drawBranch(KdNTree<T, KdNode4 > * tree, int branchIdx, 
					const BoundingBox & lftBox,
					const BoundingBox & rgtBox);
					
	template<typename T>
	void drawANode(KdNTree<T, KdNode4 > * tree, int branchIdx, 
					const KdNode4 * treelet, int idx, 
					const BoundingBox & box);
};

template<typename T>
void NTreeDrawer::drawTree(KdNTree<T, KdNode4 > * tree,
							const Vector3F & origin,
							const float & scaling)
{
	glPushMatrix();
	glTranslatef(origin.x, origin.y, origin.z);
	glScalef(scaling, scaling, scaling);
	
	const BoundingBox & box = tree->getBBox();
	drawBoundingBox(&box);
	
	KdNode4 * tn = tree->root();
	KdTreeNode * child = tn->node(0);
	if(child->isLeaf() ) {}
	else {
		const int axis = child->getAxis();
		const float pos = child->getSplitPos();
		BoundingBox lft, rgt;
		box.split(axis, pos, lft, rgt);
		drawBranch<T>(tree, tn->internalOffset(0), lft, rgt );
	}
	
	glPopMatrix();
}

template<typename T>
void NTreeDrawer::drawBranch(KdNTree<T, KdNode4 > * tree, int branchIdx, 
							const BoundingBox & lftBox,
							const BoundingBox & rgtBox)
{
	const KdNode4 * tn = tree->branches()[branchIdx];
/// first two
	drawANode<T>(tree, branchIdx, tn, 0, lftBox );
	drawANode<T>(tree, branchIdx, tn, 1, rgtBox );
}

template<typename T>
void NTreeDrawer::drawANode(KdNTree<T, KdNode4 > * tree, int branchIdx, 
					const KdNode4 * treelet, int idx, 
					const BoundingBox & box)
{
	drawBoundingBox(&box);
	const KdTreeNode * node = treelet->node(idx);
	if(node->isLeaf() ) { return; }
	const int axis = node->getAxis();
	const float pos = node->getSplitPos();
	BoundingBox lft, rgt;
	box.split(axis, pos, lft, rgt);
	
	int offset = treelet->internalOffset(idx);

/// within treelet
	if(node->getOffset() < KdNode4::TreeletOffsetMask ) {
		drawANode<T>(tree, branchIdx, treelet, idx + offset, lft );
		drawANode<T>(tree, branchIdx, treelet, idx + offset + 1, rgt );
	}
	else {
		drawBranch<T>(tree, branchIdx + offset, lft, rgt );
	}
}

}