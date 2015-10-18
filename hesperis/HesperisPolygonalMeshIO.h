#include "HesperisIO.h"
#include <map>
#include <string>
class MFnMesh;
class MIntArray;
class APolygonalMesh;
class APolygonalUV;
class MDagPath;

class HesperisPolygonalMeshCreator {
public:
    static MObject create(APolygonalMesh * data, MObject & parentObj,
                       const std::string & nodeName);
	static void addUV(APolygonalUV * data, MFnMesh & fmesh,
						const std::string & setName,
						const MIntArray & uvCounts);
	static bool checkMeshNv(const MObject & node, unsigned nv);
};

class HesperisPolygonalMeshIO : public HesperisIO {
public:
    static bool WritePolygonalMeshes(const MDagPathArray & paths, HesperisFile * file);
    static bool CreateMeshData(APolygonalMesh * data, const MDagPath & path);
	static bool CreateMeshUV(APolygonalUV * data, const MDagPath & path, const MString & setName);
    static bool ReadMeshes(MObject &target = MObject::kNullObj);
};
