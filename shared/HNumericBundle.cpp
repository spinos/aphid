#include "HNumericBundle.h"
#include <AllHdf.h>
#include <AAttribute.h>

namespace aphid {

HNumericBundle::HNumericBundle(const std::string & path): HBase(path)
{}

HNumericBundle::~HNumericBundle()
{}
	
char HNumericBundle::verifyType()
{
    if(!hasNamedAttr(".bundle_num_typ"))
		return 0;

	if(!hasNamedAttr(".bundle_sz"))
		return 0;
		
	return 1;
}

char HNumericBundle::save(const ABundleAttribute * d)
{
    if(!hasNamedAttr(".longname"))
		addStringAttr(".longname", d->longName().size());
	writeStringAttr(".longname", d->longName());
	
	int sz = d->size();
	if(!hasNamedAttr(".bundle_sz"))
		addIntAttr(".bundle_sz");
	writeIntAttr(".bundle_sz", &sz);
	
	int nt = d->numericType();
	if(!hasNamedAttr(".bundle_num_typ"))
		addIntAttr(".bundle_num_typ");

	writeIntAttr(".bundle_num_typ", &nt);
	
	int l = d->dataLength();
	if(!hasNamedData(".raw"))
	    addCharData(".raw", l);
		
	writeCharData(".raw", l, (char *)d->value());

    return 1;
}

char HNumericBundle::load(ABundleAttribute * d)
{
/*
	int nv = 3;
	readIntAttr(".nv", &nv);
	int npoly = 1;
	readIntAttr(".npoly", &npoly);
	int nfv = 3;
	readIntAttr(".ninds", &nfv);
	poly->create(nv, nfv, npoly);
	
	readVector3Data(".p", poly->numPoints(), (Vector3F *)poly->points());
	readIntData(".a", poly->numPoints(), (unsigned *)poly->anchors());
	readIntData(".v", poly->numIndices(), (unsigned *)poly->indices());
	readIntData(".fcnt", poly->numPolygons(), (unsigned *)poly->faceCounts());
	readIntData(".fdft", poly->numPolygons(), (unsigned *)poly->faceDrifts());
*/	
    return 1;
}

}
