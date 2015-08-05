#include "HField.h"
#include <AllHdf.h>
#include <SHelper.h>

HField::HField(const std::string & path) : HBase(path) {}
HField::~HField() {}
	
char HField::verifyType() 
{
    if(!hasNamedAttr(".fieldNumChannels"))
        return 0;
    
    if(!hasNamedAttr(".fieldChannelNames"))
        return 0;
    
    if(!hasNamedAttr(".fieldNumElements"))
        return 0;
    return 1;
}

char HField::save(AField * fld) 
{
    std::vector<std::string > names;
    fld->getChannelNames(names);
    
    const int nc = names.size();
    if(!hasNamedAttr(".fieldNumChannels"))
		addIntAttr(".fieldNumChannels");
	
	writeIntAttr(".fieldNumChannels", &nc);
    
    
    return 1;
}

char HField::load(AField * fld) 
{
    if(!verifyType()) return 0;
    
    
    return 1;
}
