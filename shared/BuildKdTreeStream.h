/*
 *  BuildKdTreeStream.h
 *  kdtree
 *
 *  Created by jian zhang on 10/29/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BoundingBox.h>
#include <VectorArray.h>
#include <IndexArray.h>
#include <KdTreeNodeArray.h>
#include <BaseMesh.h>
#include <PatchMesh.h>
class BuildKdTreeStream {
public:
	BuildKdTreeStream();
	~BuildKdTreeStream();
	void initialize();
	void cleanup();
	void appendGeometry(Geometry * geo);
	
	const unsigned getNumPrimitives() const;
	const sdb::VectorArray<Primitive> &getPrimitives() const;

	sdb::VectorArray<Primitive> &primitives();
	IndexArray &indirection();
	
	KdTreeNode *createTreeBranch();
	KdTreeNode *firstTreeBranch();
	
	void verbose() const;
	
private:
	sdb::VectorArray<Primitive> m_primitives;
	IndexArray m_indirection;
	KdTreeNodeArray m_nodes;
};