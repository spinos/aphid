/*
 *  MeshSeparator.h
 *  aphid
 *
 *  Created by jian zhang on 8/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <Array.h>
#include <vector>
class ATriangleMesh;
class MeshSeparator {
public:
	MeshSeparator();
	virtual ~MeshSeparator();
	
	void separate(ATriangleMesh * m);
	unsigned numPatches() const;
	unsigned patchTriangleBegin(unsigned i) const;
protected:
typedef sdb::Array<unsigned, char> VertexIndices;

	bool isVerticesConnectedToLastPatch(unsigned * v);
	void connectVerticesToLastPatch(unsigned * v);
	void clearLastPatch();
private:
	VertexIndices m_lastPatch;
	std::vector<unsigned> m_patchTriangleBegins;
};