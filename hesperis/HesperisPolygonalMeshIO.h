#include "HesperisIO.h"
class APolygonalMesh;
class APolygonalUV;
class MDagPath;

class HesperisPolygonalMeshCreator {
public:
    static MObject create(APolygonalMesh * data, MObject & parentObj,
                       const std::string & nodeName);
};

class HesperisPolygonalMeshIO : public HesperisIO {
public:
    static bool WritePolygonalMeshes(MDagPathArray & paths, HesperisFile * file);
    static bool CreateMeshData(APolygonalMesh * data, const MDagPath & path);
	static bool CreateMeshUV(APolygonalUV * data, const MDagPath & path, const MString & setName);
    static bool ReadMeshes(HesperisFile * file, MObject &target = MObject::kNullObj);
};
