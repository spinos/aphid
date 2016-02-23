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

namespace aphid {

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
    readIntAttr(".attrib_typ", &m_savedType);
	return 1;
}

int HAttributeEntry::savedType() const
{ return m_savedType; }

HIntAttributeEntry::HIntAttributeEntry(const std::string & path) : HAttributeEntry(path) {} 
HIntAttributeEntry::~HIntAttributeEntry() {}

HAttributeEntry::AttributeType HIntAttributeEntry::attributeType() const
{ return tInt; }
	
char HIntAttributeEntry::save(int * src)
{
    if(!hasNamedAttr(".def_val"))
        addIntAttr(".def_val");
    
    writeIntAttr(".def_val", src);
    return HAttributeEntry::save();
}

char HIntAttributeEntry::load(int * dst)
{
    readIntAttr(".def_val", dst);
    return 1;
}

HFltAttributeEntry::HFltAttributeEntry(const std::string & path) : HAttributeEntry(path) {} 
HFltAttributeEntry::~HFltAttributeEntry() {}
	
HAttributeEntry::AttributeType HFltAttributeEntry::attributeType() const
{ return tFlt; }

char HFltAttributeEntry::save(float * src)
{
    if(!hasNamedAttr(".def_val"))
        addFloatAttr(".def_val");
    
    writeFloatAttr(".def_val", src);
    return HAttributeEntry::save();
}

char HFltAttributeEntry::load(float * dst)
{
    readFloatAttr(".def_val", dst);
    return 1;
}

HFlt3AttributeEntry::HFlt3AttributeEntry(const std::string & path) : HAttributeEntry(path) {} 
HFlt3AttributeEntry::~HFlt3AttributeEntry() {}
	
HAttributeEntry::AttributeType HFlt3AttributeEntry::attributeType() const
{ return tFlt3; }
	
char HFlt3AttributeEntry::save(const Vector3F * src)
{
    if(!hasNamedAttr(".def_val"))
        addFloatAttr(".def_val", 3);
    
    writeFloatAttr(".def_val", (float *)src);
    return HAttributeEntry::save();
}

char HFlt3AttributeEntry::load(Vector3F * dst)
{
    readFloatAttr(".def_val", (float *)dst);
    return 1;
}

}
//:~