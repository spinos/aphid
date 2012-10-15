/*
 *  KdTree.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "KdTree.h"
#include <Primitive.h>

KdTree::KdTree() 
{
	m_root = new KdTreeNode;
}

KdTree::~KdTree() 
{
	delete m_root;
}

KdTreeNode* KdTree::GetRoot() { return m_root; }

void KdTree::create(BaseMesh* mesh)
{printf("prim sz %d\n", (int)sizeof(Primitive));


	unsigned nf = mesh->getNumFaces();
	printf("num triangles %i \n", nf);
	
	typedef Primitive * primitivePtr;
	primitivePtr * primitives = new primitivePtr[nf];
	for(unsigned i = 0; i < nf; i++) {
		primitives[i] = mesh->getFace(i);
		//primitives[i]->name();
	}
		
	for(unsigned i = 0; i < nf; i++) delete primitives[i];
	delete[] primitives;
}