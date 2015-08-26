#include "HTransform.h"
#include "BaseTransform.h"
#include "Matrix44F.h"
HTransform::HTransform(const std::string & path) : HBase(path) 
{}

HTransform::~HTransform()
{}
	
char HTransform::verifyType()
{
    if(!hasNamedAttr(".transform_type"))
		return 0;
    return 1;
}

char HTransform::save()
{
    if(!hasNamedAttr(".transform_type"))
        addIntAttr(".transform_type");
    
	int t = 0;
    writeIntAttr(".transform_type", &t);
	
	if(!hasNamedAttr(".translate"))
		addFloatAttr(".translate", 3);
		
	float zero3[3] = {0.f, 0.f, 0.f};
	writeFloatAttr(".translate", zero3);
	
	if(!hasNamedAttr(".rotate"))
		addFloatAttr(".rotate", 3);
		
	writeFloatAttr(".rotate", zero3);
	
	if(!hasNamedAttr(".scale"))
		addFloatAttr(".scale", 3);
		
	float one3[3] = {1.f, 1.f, 1.f};
	writeFloatAttr(".scale", one3);
    return 1;
}

char HTransform::save(BaseTransform * tm)
{
    if(!hasNamedAttr(".transform_type"))
        addIntAttr(".transform_type");
// todo differentiate type
	int t = 0;
    writeIntAttr(".transform_type", &t);
	
	if(!hasNamedAttr(".translate"))
		addFloatAttr(".translate", 3);
		
    Vector3F v3 = tm->translation();
	writeFloatAttr(".translate", (float *)&v3);
	
    if(!hasNamedAttr(".rotate"))
		addFloatAttr(".rotate", 3);
		
	v3 = tm->rotationAngles();
	writeFloatAttr(".rotate", (float *)&v3);
	
	if(!hasNamedAttr(".scale"))
		addFloatAttr(".scale", 3);
		
	v3 = tm->scale();
	writeFloatAttr(".scale", (float *)&v3);
	
    return 1;
}

char HTransform::load(BaseTransform * tm)
{
	Vector3F t3;
	if(hasNamedAttr(".translate")) {
		readFloatAttr(".translate", (float *)&t3);
		tm->setTranslation(t3);
	}
	
	if(hasNamedAttr(".rotate")) {
		readFloatAttr(".rotate", (float *)&t3);
		tm->setRotationAngles(t3);
	}
	
	if(hasNamedAttr(".scale")) {
		readFloatAttr(".scale", (float *)&t3);
		tm->setScale(t3);
	}
    return 1;
}
//:~