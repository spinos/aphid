#include "HPolygonalUV.h"
#include <APolygonalUV.h>

HPolygonalUV::HPolygonalUV(const std::string & path) : HBase(path) 
{}

HPolygonalUV::~HPolygonalUV() 
{}

char HPolygonalUV::verifyType()
{
	if(!hasNamedAttr(".nuv"))
		return 0;

	if(!hasNamedAttr(".nind"))
		return 0;
	
	return 1;
}

char HPolygonalUV::save(APolygonalUV * poly)
{
	int nuv = poly->numCoords();
	if(!hasNamedAttr(".nuv"))
		addIntAttr(".nuv");
	
	writeIntAttr(".nuv", &nuv);
	
	int nind = poly->numIndices();
	if(!hasNamedAttr(".nind"))
		addIntAttr(".nind");
	
	writeIntAttr(".nind", &nind);
	
	if(!hasNamedData(".ucoord"))
	    addFloatData(".ucoord", nuv);
	
	writeFloatData(".ucoord", nuv, (float *)poly->ucoord());
	
	if(!hasNamedData(".vcoord"))
	    addFloatData(".vcoord", nuv);
	
	writeFloatData(".vcoord", nuv, (float *)poly->vcoord());
		
	if(!hasNamedData(".uvid"))
	    addIntData(".uvid", nind);
	
	writeIntData(".uvid", nind, (int *)poly->indices());

	return 1;
}

char HPolygonalUV::load(APolygonalUV * poly)
{
	int nuv = 3;
	
	readIntAttr(".nuv", &nuv);
	
	int nind = 1;
	
	readIntAttr(".nind", &nind);
	
	poly->create(nuv, nind);
	
	readFloatData(".ucoord", poly->numCoords(), poly->ucoord());
	readFloatData(".vcoord", poly->numCoords(), poly->vcoord());
	readIntData(".uvid", poly->numIndices(), poly->indices());
	
	return 1;
}
