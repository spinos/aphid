#include "HesperisIO.h"
class APolygonalMesh;
class 
class MDagPath;

class HesperisPolygonalMeshIO : public HesperisIO {
public:
    static bool WritePolygonalMeshes(MDagPathArray & paths, HesperisFile * file);
    static bool CreateMeshData(APolygonalMesh * data, const MDagPath & path);
	static bool CreateMeshUV(APolygonalUV * data, const MDagPath & path, const MString & setName);
};
