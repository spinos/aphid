#include <geom/ATetrahedronMeshGroup.h>

namespace aphid {

ATetrahedronMeshGroup::ATetrahedronMeshGroup() {}
ATetrahedronMeshGroup::~ATetrahedronMeshGroup() {}
	
void ATetrahedronMeshGroup::create(unsigned np, unsigned nt, unsigned ns)
{
	ATetrahedronMesh::create(np, nt);
	AStripedModel::create(ns);
}

std::string ATetrahedronMeshGroup::verbosestr() const
{
	std::stringstream sst;
	sst<<ATetrahedronMesh::verbosestr()
    <<" nstripe "<<numStripes()
    <<"\n";
	return sst.str();
}

}

