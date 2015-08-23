/*
 *  MeshSeparator.h
 *  aphid
 *
 *  Created by jian zhang on 8/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <Array.h>
#include <map>
#include <vector>
class BaseBuffer;
class ATriangleMesh;
class MeshSeparator {
public:
	MeshSeparator();
	virtual ~MeshSeparator();
	
	void separate(ATriangleMesh * m);
	unsigned numPatches() const;
	
	void patchBegin();
	bool patchEnd();
	void nextPatch();
	unsigned getPatchCvs(BaseBuffer * pos, ATriangleMesh * m);
protected:
typedef sdb::Array<unsigned, char> VertexIndices;

	bool isVerticesConnectedToAnyPatch(unsigned * v, int & ipatch);
	void connectVerticesToPatch(unsigned * v, unsigned ipatch);
	void addPatch(unsigned x);
	void clearPatches();
	void mergeConnectedPatches();
	bool isPatchConnectedToAnyPatch(unsigned t, unsigned & connectedPatch, VertexIndices * v);
	void mergePatches(unsigned a, unsigned b);
	void removePatch(unsigned a);
private:
	std::map<unsigned, VertexIndices *> m_patches;
	std::map<unsigned, VertexIndices *>::iterator m_patchIt;
};