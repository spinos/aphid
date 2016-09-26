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
		
    if(!hasNamedAttr(".longname"))
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
	int nt = 0;
	readIntAttr(".bundle_num_typ", &nt);
	
	int sz = 0;
	readIntAttr(".bundle_sz", &sz);
	
	ANumericAttribute::NumericAttributeType dt = ANumericAttribute::TUnkownNumeric;
	if( nt == ANumericAttribute::TByteNumeric ) {
		dt = ANumericAttribute::TByteNumeric;
	}
	else if( nt == ANumericAttribute::TShortNumeric ) {
		dt = ANumericAttribute::TShortNumeric;
	}
	else if( nt == ANumericAttribute::TIntNumeric ) {
		dt = ANumericAttribute::TIntNumeric;
	}
	else if( nt == ANumericAttribute::TFloatNumeric ) {
		dt = ANumericAttribute::TIntNumeric;
	}
	else if( nt == ANumericAttribute::TDoubleNumeric ) {
		dt = ANumericAttribute::TDoubleNumeric;
	}
	else if( nt == ANumericAttribute::TBooleanNumeric ) {
		dt = ANumericAttribute::TBooleanNumeric;
	}
	
	if(dt == ANumericAttribute::TUnkownNumeric)
	    return 0;
	
	d->create(sz, dt);
	
	int l = d->dataLength();
	readCharData(".raw", l, d->value() );
	
	std::string lnm;
	readStringAttr(".longname", lnm);
	
	d->setLongName(lnm);
	d->setShortName(lastName());

    return 1;
}

}
