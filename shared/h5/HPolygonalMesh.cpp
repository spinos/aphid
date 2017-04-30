#include "HPolygonalMesh.h"
#include <AllHdf.h>
#include <geom/APolygonalMesh.h>
#include <geom/APolygonalUV.h>
#include <h5/HPolygonalUV.h>

namespace aphid {

HPolygonalMesh::HPolygonalMesh(const std::string & path): HBase(path)
{}

HPolygonalMesh::~HPolygonalMesh()
{}
	
char HPolygonalMesh::verifyType()
{
    if(!hasNamedAttr(".npoly"))
		return 0;

	if(!hasNamedAttr(".nv"))
		return 0;
		
	if(!hasNamedData(".fcnt"))
		return 0;
		
	return 1;
}

char HPolygonalMesh::save(APolygonalMesh * poly)
{
	int nv = poly->numPoints();
	if(!hasNamedAttr(".nv"))
		addIntAttr(".nv");
	writeIntAttr(".nv", &nv);
	
	int nf = poly->numPolygons();
	if(!hasNamedAttr(".npoly"))
		addIntAttr(".npoly");

	writeIntAttr(".npoly", &nf);
	
	int nfv = poly->numIndices();
	if(!hasNamedAttr(".ninds"))
		addIntAttr(".ninds");
		
	writeIntAttr(".ninds", &nfv);
	
	if(!hasNamedData(".p"))
	    addVector3Data(".p", nv);
	
	writeVector3Data(".p", nv, (Vector3F *)poly->points());
	
	if(!hasNamedData(".a"))
	    addIntData(".a", nv);
	
	writeIntData(".a", nv, (int *)poly->anchors());
	
	if(!hasNamedData(".v"))
	    addIntData(".v", nfv);
	
	writeIntData(".v", nfv, (int *)poly->indices());
	
	if(!hasNamedData(".fcnt"))
	    addIntData(".fcnt", nf);
		
	writeIntData(".fcnt", nf, (int *)poly->faceCounts());
	
	if(!hasNamedData(".fdft"))
	    addIntData(".fdft", nf);
		
	writeIntData(".fdft", nf, (int *)poly->faceDrifts());
	
	unsigned i = 0;
	for(;i<poly->numUVs();i++) {
		std::string uvName = poly->uvName(i);
		HPolygonalUV grpUV(childPath(uvName));
		grpUV.save(poly->uvData(uvName));
		grpUV.close();
	}
	
    return 1;
}

char HPolygonalMesh::load(APolygonalMesh * poly)
{
	int nv = 3;
	readIntAttr(".nv", &nv);
	int npoly = 1;
	readIntAttr(".npoly", &npoly);
	int nfv = 3;
	readIntAttr(".ninds", &nfv);
	poly->create(nv, nfv, npoly);
	
	readVector3Data(".p", poly->numPoints(), (Vector3F *)poly->points());
	readIntData(".a", poly->numPoints(), (int *)poly->anchors());
	readIntData(".v", poly->numIndices(), (int *)poly->indices());
	readIntData(".fcnt", poly->numPolygons(), (int *)poly->faceCounts());
	readIntData(".fdft", poly->numPolygons(), (int *)poly->faceDrifts());
	
	std::vector<std::string > uvNames;
	lsTypedChild<HPolygonalUV>(uvNames);
	
	std::vector<std::string >::const_iterator it = uvNames.begin();
	for(;it!=uvNames.end();++it) {
		APolygonalUV * auv = new APolygonalUV;
		HPolygonalUV grpUV(*it);
		grpUV.load(auv);
		grpUV.close();
		poly->addUV(*it, auv);
	}
    return 1;
}

}