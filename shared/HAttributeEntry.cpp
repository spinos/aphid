/*
 *  HAttributeEntry.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HAttributeEntry.h"
#include <AllHdf.h>

HAttributeEntry::HAttributeEntry(const std::string & path) : HBase(path) {}
HAttributeEntry::~HAttributeEntry() {}

HAttributeEntry::AttributeType HAttributeEntry::attributeType() const
{ return tUnknown; }

char HAttributeEntry::verifyType()
{
	if(!hasNamedAttr(".attrib_typ"))
		return 0;

	if(!hasNamedAttr(".def_val"))
		return 0;
	
	return 1;
}

char HAttributeEntry::save() 
{
	if(!hasNamedAttr(".attrib_typ"))
		addIntAttr(".attrib_typ");
	
	int typasi = attributeType();
	writeIntAttr(".attrib_typ", &typasi);
	return 1;
}

char HAttributeEntry::load()
{
	return 1;
}
//:~