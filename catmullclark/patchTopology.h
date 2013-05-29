/*
 *  patchTopology.h
 *  catmullclark
 *
 *  Created by jian zhang on 10/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
class Vector3F;
class PatchTopology 
{
public:
	PatchTopology();
	~PatchTopology();
	
	void setVertexValence(unsigned* data);
	void setVertex(unsigned* data);
	void setBoundary(char* data);
	
	int getCornerIndex(int i) const;
	char isCornerOnBoundary(int i) const;
	char isEdgeOnBoundary(int i) const;
	void getBoundaryEdgesOnCorner(int i, int* edge) const;
	void getFringeAndEdgesOnCorner(int i, int* fringe, int* edge, char &cornerBehind) const;
	void getFringeAndEdgesOnEdgeBySide(int i, int side, int* fringe, int* edge) const;
	void getFringeOnFaceByCorner(int i, int* fringe) const;
	void getEdgeBySide(int i, int side, int* edge) const;
	int getValenceOnCorner(int i) const;
	int getCornerValenceByEdgeBySide(int i, int side) const;
	
private:
	unsigned* _valence;
	unsigned* _vertices;
	char* _boundary;
	int V(int i) const;
};