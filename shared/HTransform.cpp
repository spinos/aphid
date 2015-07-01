#include "HTransform.h"
#include "BaseTransform.h"
#include "Matrix44F.h"
HTransform::HTransform(const std::string & path) : HBase(path) 
{}

HTransform::~HTransform()
{}
	
char HTransform::verifyType()
{
    if(!hasNamedData(".tm"))
		return 0;
    return 1;
}

char HTransform::save()
{
    if(!hasNamedData(".tm"))
        addFloatData(".tm", 16);
    
    Matrix44F mat;
    mat.setIdentity();
    writeFloatData(".tm", 16, (float *)mat.v);
    return 1;
}

char HTransform::save(BaseTransform * tm)
{
    if(!hasNamedData(".tm"))
        addFloatData(".tm", 16);
    
    Matrix44F mat = tm->space();
    writeFloatData(".tm", 16, (float *)mat.v);
    return 1;
}

char HTransform::load(BaseTransform * tm)
{
    return 1;
}
