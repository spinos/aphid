#include "EasyModelIn.h"
#include <fstream>
EasyModelIn::EasyModelIn(const char* filename) : _data(0)
{
	if(_doc.load(filename) != 1)
	{
		printf("cannot load %s\n", filename);
		return;
	}	
	int ndata = _doc.getIntAttribByName("data_size"); 
	_data = new char[ndata];
	if(!loadBinFile(_doc._file_name.c_str(), _data, ndata))
	{
		printf("cannot load data for %s\n", filename);
	}
	if(!findFirstMesh())
	{
		printf("cannot find any mesh\n");
		return;
	}
	readFaceCount(); 
	readFaceConnection();
	readP();
	readPatchVertex();
	readPatchBoundary();
	readVertexValence();
	readVertexNormal();
	readUV();
	printf("model successfully loaded from %s\n", filename);
}

EasyModelIn::~EasyModelIn()
{
	if(_data) delete[] _data;
}

unsigned EasyModelIn::getNumFace() const
{
	return _numPolygon;
}

unsigned EasyModelIn::getNumVertex() const
{
	return _numVertex;
}

unsigned EasyModelIn::getNumUVs() const
{
	return _numUVs;
}

unsigned EasyModelIn::getNumUVIds() const
{
	return _numUVIds;
}

int* EasyModelIn::getFaceCount() const
{
	return _faceCount;
}

int* EasyModelIn::getFaceConnection() const
{
	return _vertices;
}

float* EasyModelIn::getVertexPosition() const
{
	return _cvs;
}

float* EasyModelIn::getVertexNormal() const
{
	return _normals;
}

int* EasyModelIn::getVertexValence() const
{
	return _vertexValence;
}

int* EasyModelIn::getUVIds() const
{
	return _uvVertices;
}

float* EasyModelIn::getUs() const
{
	return _us;
}

float* EasyModelIn::getVs() const
{
	return _vs;
}

int* EasyModelIn::getPatchVertex() const
{
	return _patchVertex;
}
	
char* EasyModelIn::getPatchBoundary() const
{
	return _patchBoundary;
}

void EasyModelIn::setPatchAtFace(unsigned index, unsigned* vertex, char* boundary) const
{
	int offset = index * 24;
	for(int i = 0; i < 24; i++)
		vertex[i] = _patchVertex[offset + i];
		
	offset = index * 15;
	for(int i = 0; i < 15; i++)
		boundary[i] = _patchBoundary[offset + i];
}

char EasyModelIn::loadBinFile(const char* filename, char *data, int count)
{
	std::string bin_path = filename;
	int found = bin_path.rfind('.', bin_path.size());
	if(found > 1) 
	{
		bin_path.erase(found);
	}	
	bin_path.append(".bm");

	std::ifstream ffin;
	ffin.open(bin_path.c_str(), std::ios::in | std::ios::binary);
	if(!ffin.is_open())
		return 0;
		
	ffin.read((char*)data, count);
	return 1;
}
          
char EasyModelIn::findFirstMesh()
{
	_doc.setChildren();
	char hasNext = 1;
	while(hasNext)
	{
		if(_doc.checkNodeName("mesh") == 1)
		{
			printf("found a mesh %s\n", _doc.getAttribByName("name"));
			return 1;
		}
		hasNext = _doc.nextNode();
	}
	return 0;
}

void EasyModelIn::readFaceCount()
{
	_doc.getChildByName("faceCount");
	_numPolygon = _doc.getIntAttribByName("count");
	int pos = _doc.getIntAttribByName("loc");

	char *ptr = _data;
	ptr += pos;
	_faceCount = (int*)ptr;
	_doc.setParent();
}

void EasyModelIn::readFaceConnection()
{
	_doc.getChildByName("faceConnection");
	int pos = _doc.getIntAttribByName("loc");

	char *ptr = _data;
	ptr += pos;
	_vertices = (int*)ptr;
	_doc.setParent();
}

void EasyModelIn::readP()
{
	_doc.getChildByName("P");
	_numVertex = _doc.getIntAttribByName("count");
	int pos = _doc.getIntAttribByName("loc");

	char *ptr = _data;
	ptr += pos;
	_cvs = (float*)ptr;
	_doc.setParent();
}

void EasyModelIn::readPatchVertex()
{
	_doc.getChildByName("PatchVertex");
	int pos = _doc.getIntAttribByName("loc");

	char *ptr = _data;
	ptr += pos;
	_patchVertex = (int*)ptr;
	_doc.setParent();
}

void EasyModelIn::readPatchBoundary()
{
	_doc.getChildByName("PatchBoundary");
	int pos = _doc.getIntAttribByName("loc");

	char *ptr = _data;
	ptr += pos;
	_patchBoundary = ptr;
	_doc.setParent();
}

void EasyModelIn::readVertexValence()
{
	_doc.getChildByName("VertexValence");
	int pos = _doc.getIntAttribByName("loc");

	char *ptr = _data;
	ptr += pos;
	_vertexValence = (int*)ptr;
	_doc.setParent();
}

void EasyModelIn::readVertexNormal()
{
	_doc.getChildByName("VertexNormal");
	int pos = _doc.getIntAttribByName("loc");

	char *ptr = _data;
	ptr += pos;
	_normals = (float*)ptr;
	_doc.setParent();
}

void EasyModelIn::readUV()
{
	_doc.getChildByName("uvSet");
	_doc.getChildByName("s");
	_numUVs = _doc.getIntAttribByName("count");
	
	int pos = _doc.getIntAttribByName("loc");
	
	char *ptr = _data;
	ptr += pos;
	_us = (float*)ptr;
	_doc.setParent();
	
	_doc.getChildByName("t");
	pos = _doc.getIntAttribByName("loc");
	ptr = _data;
	ptr += pos;
	_vs = (float*)ptr;
	_doc.setParent();
	
	_doc.getChildByName("uvid");
	_numUVIds = _doc.getIntAttribByName("count");
	pos = _doc.getIntAttribByName("loc");
	ptr = _data;
	ptr += pos;
	_uvVertices = (int*)ptr;
	_doc.setParent();
	
	_doc.setParent();
}