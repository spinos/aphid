#include "HElemAsset.h"

namespace aphid {

HElemBase::HElemBase(const std::string & name) :
HBase(name)
{}

HElemBase::~HElemBase()
{}

char HElemBase::verifyType()
{
	if(!hasNamedAttr(".bbx") ) return 0;
	if(!hasNamedAttr(".elemtyp") ) return 0;
	if(!hasNamedAttr(".nelem") ) return 0;
	if(!hasNamedData(".data") ) return 0;
	return 1;
}

}