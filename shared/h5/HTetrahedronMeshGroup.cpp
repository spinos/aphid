#include "HTetrahedronMeshGroup.h"
#include "geom/ATetrahedronMeshGroup.h"

namespace aphid {

HTetrahedronMeshGroup::HTetrahedronMeshGroup(const std::string & path)
    : HTetrahedronMesh(path) 
{}
    
HTetrahedronMeshGroup::~HTetrahedronMeshGroup() {}
	
char HTetrahedronMeshGroup::verifyType()
{
    if(!hasNamedAttr(".npart"))
		return 0;
		
	return HTetrahedronMesh::verifyType();
}

char HTetrahedronMeshGroup::save(ATetrahedronMeshGroup * tetra)
{
    if(!hasNamedAttr(".npart"))
		addIntAttr(".npart");
		
	int np = tetra->numStripes();
	writeIntAttr(".npart", &np);
	
	if(!hasNamedData(".pntdrift"))
		addIntData(".pntdrift", np);
		
	writeIntData(".pntdrift", np, (int *)tetra->pointDrifts());
	
	if(!hasNamedData(".inddrift"))
		addIntData(".inddrift", np);
		
	writeIntData(".inddrift", np, (int *)tetra->indexDrifts());
	
	return HTetrahedronMesh::save(tetra);
}

char HTetrahedronMeshGroup::load(ATetrahedronMeshGroup * tetra)
{
    int npart = 1;
	readIntAttr(".npart", &npart);
	
	int nv = 3;
	readIntAttr(".nv", &nv);
	
	int nt = 1;
	readIntAttr(".nt", &nt);
	
	tetra->create(nv, nt, npart);

	readIntData(".pntdrift", npart, (unsigned *)tetra->pointDrifts());
	readIntData(".inddrift", npart, (unsigned *)tetra->indexDrifts());
	
	return HTetrahedronMesh::readAftCreation(tetra);
}

}