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

	sdb::VectorArray<Primitive> &primitives();
	sdb::VectorArray<Primitive> &indirection();
	
	KdTreeNode *createTreeBranch();
	KdTreeNode *firstTreeBranch();
	
	const unsigned & numNodes() const;
	
	void verbose() const;
	BoundingBox calculateComponentBox(const int & igeom, const int & icomp);
	Geometry * geometry(const int & igeom);
	unsigned numGeometries() const;
	unsigned numIndirections() const;
	void removeInput();
	
private:
	sdb::VectorArray<Primitive> m_primitives;
	sdb::VectorArray<Primitive> m_indirection;
	sdb::VectorArray<KdTreeNode> m_nodeBlks;
	std::vector<Geometry *> m_geoms;
};

}