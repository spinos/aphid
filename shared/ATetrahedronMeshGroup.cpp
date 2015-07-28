#include "ATetrahedronMeshGroup.h"

ATetrahedronMeshGroup::ATetrahedronMeshGroup() {}
ATetrahedronMeshGroup::~ATetrahedronMeshGroup() {}
	
void ATetrahedronMeshGroup::create(unsigned np, unsigned nt, unsigned ns)
{
	ATetrahedronMesh::create(np, nt);
	AStripedModel::create(ns);
}

