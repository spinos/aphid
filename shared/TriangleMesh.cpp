#include "TriangleMesh.h"
#include "modelIn.h"
TriangleMesh::TriangleMesh() {}

TriangleMesh::TriangleMesh(const char * filename)
{
    EasyModel * esm = new EasyModel("D:/aphid/lapl/simple.m");
	copyOf(esm);
	delete esm;
}

TriangleMesh::~TriangleMesh() {}

void TriangleMesh::copyOf(EasyModel * esm)
{
    const unsigned nf = esm->getNumFace();
    
    int *faceCount = esm->getFaceCount();
    int *faceConnection = esm->getFaceConnection();

    _numFaces = 0;
    
    unsigned i, j;
    for(i = 0; i < nf; i++)
        _numFaces += faceCount[i] - 2;
    
    _numFaceVertices = _numFaces * 3;
    
    _indices = new unsigned[_numFaceVertices];
    
    unsigned curTri = 0;
    unsigned curFace = 0;
    for(i = 0; i < nf; i++) {
        for(j = 1; j < faceCount[i] - 1; j++) {
            _indices[curTri] = faceConnection[curFace];
            _indices[curTri + 1] = faceConnection[curFace + j];
            _indices[curTri + 2] = faceConnection[curFace + j + 1];
            curTri += 3;
        }
        curFace += faceCount[i];
    }
    
    float* cvs = esm->getVertexPosition();
    _numVertices = esm->getNumVertex();
    _vertices = new Vector3F[_numVertices];
    
    for(i = 0; i < _numVertices; i++) {
        _vertices[i].x = cvs[i * 3];
        _vertices[i].y = cvs[i * 3 + 1];
        _vertices[i].z = cvs[i * 3 + 2];
    }
}
