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
#include <BaseMesh.h>
#include <KdTreeNode.h>
#include <vector>

namespace aphid {

class BuildKdTreeStream {

	unsigned m_numNodes;
	KdTreeNode * m_nodeBuf;
	
public:
	BuildKdTreeStream();
	~BuildKdTreeStream();
	void initialize();
	void cleanup();
	void appendGeometry(Geometry * geo);
	
	const unsigned getNumPrimitives() const;
	const sdb::VectorArray<Primitive> &getPrimitives() const;

	sdb::VectorArray<Primitive> &primitives();
	sdb::VectorArray<unsigned> &indirection();
	
	KdTreeNode *createTreeBranch();
	KdTreeNode *firstTreeBranch();
	
	const unsigned & numNodes() const;
	
	void verbose() const;
	
private:
	sdb::VectorArray<Primitive> m_primitives;
	sdb::VectorArray<unsigned> m_indirection;
	sdb::VectorArray<KdTreeNode> m_nodeBlks;
};

}