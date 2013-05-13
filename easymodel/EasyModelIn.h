#ifndef EasyModelInIN_H
#define EasyModelInIN_H
#include <ZXMLDoc.h>

class EasyModelIn {
public:
	EasyModelIn(const char* filename);
	~EasyModelIn();
	unsigned getNumFace() const;
	unsigned getNumVertex() const;
	unsigned getNumValence() const;
	int* getFaceCount() const;
	int* getFaceConnection() const;
	float* getVertexPosition() const;
	float* getVertexNormal() const;
	int* getVertexValence() const;
	int* getUVIds() const;
	float* getUs() const;
	float* getVs() const;
	int* getPatchVertex() const;
	char* getPatchBoundary() const;
	void setPatchAtFace(unsigned index, unsigned* vertex, char* boundary) const;
	
private:
	unsigned _numPolygon;
	unsigned _numVertex;
	unsigned _numValence;
	int* _faceCount;
	int* _vertices;
	float* _cvs;
	float* _normals;
	float* _us;
	float* _vs;
	int* _uvVertices;
	int* _patchVertex;
	int* _vertexValence;
	char* _patchBoundary;
	ZXMLDoc _doc;
	char* _data;
	
	char loadBinFile(const char* filename, char *data, int count);
	char findFirstMesh();
	void readFaceCount();
	void readFaceConnection();
	void readPatchVertex();
	void readPatchBoundary();
	void readVertexValence();
	void readVertexNormal();
	void readP();
	void readUV();
};
#endif        //  #ifndef EasyModelIn_H

