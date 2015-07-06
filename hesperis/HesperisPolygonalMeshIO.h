#include "HesperisIO.h"

class HesperisPolygonalMeshIO : public HesperisIO {
public:
    static bool HesperisPolygonalMeshIO::WritePolygonalMeshes(MDagPathArray & paths, HesperisFile * file);
};
