#include "HesperisIO.h"
class APolygonalMesh;
class MDagPath;

class HesperisPolygonalMeshIO : public HesperisIO {
public:
    static bool WritePolygonalMeshes(MDagPathArray & paths, HesperisFile * file);
    static bool CreateMeshData(APolygonalMesh * data, const MDagPath & path);
};
