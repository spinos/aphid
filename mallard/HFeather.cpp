/*
 *  HFeather.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HFeather.h"
#include <MlFeather.h>
HFeather::HFeather(const std::string & path) : HBase(path) {}

char HFeather::save(MlFeather * feather)
{
	feather->verbose();
	
	int id = feather->featherId();
	if(!hasNamedAttr(".featherId"))
		addIntAttr(".featherId");
		
	writeIntAttr(".featherId", &id);
	
	int nseg = feather->numSegment();
	
	if(!hasNamedAttr(".nseg"))
		addIntAttr(".nseg");
		
	writeIntAttr(".nseg", &nseg);
	
	if(!hasNamedData(".quill"))
		addFloatData(".quill", nseg);
		
	writeFloatData(".quill", nseg, feather->quilly());
	
	int nvv = feather->numVaneVertices();
	if(!hasNamedAttr(".nvv"))
		addIntAttr(".nvv");
		
	writeIntAttr(".nvv", &nvv);
	
	if(!hasNamedData(".vv"))
		addFloatData(".vv", nvv * 2);
		
	writeFloatData(".vv", nvv * 2, (float *)feather->uvDisplace());
	
	if(!hasNamedAttr(".uv"))
		addFloatAttr(".uv", 2);
		
	Vector2F uv = feather->baseUV();
	writeFloatAttr(".uv", (float *)(&uv));
	
	if(!hasNamedAttr(".typ")) addIntAttr(".typ");
	int typ = feather->type();
	writeIntAttr(".typ", &typ);
	
	if(!hasNamedAttr(".rs")) addIntAttr(".rs");
	int rs = feather->resShaft();
	writeIntAttr(".rs", &rs);
	
	if(!hasNamedAttr(".rb")) addIntAttr(".rb");
	int rb = feather->resBarb();
	writeIntAttr(".rb", &rb);
	
	if(!hasNamedAttr(".nsep")) addIntAttr(".nsep");
	int nsep = feather->numSeparate();
	writeIntAttr(".nsep", &nsep);
	
	if(!hasNamedAttr(".kfuzz")) addFloatAttr(".kfuzz");
	float kfuzz = feather->fuzzy();
	writeFloatAttr(".kfuzz", &kfuzz);
	
	if(!hasNamedAttr(".ksep")) addFloatAttr(".ksep");
	float ksep = feather->separateStrength();
	writeFloatAttr(".ksep", &ksep);
	
	if(!hasNamedAttr(".shrb")) addFloatAttr(".shrb");
	float barbShrink = feather->m_barbShrink;
	writeFloatAttr(".shrb", &barbShrink);
	
	if(!hasNamedAttr(".shrs")) addFloatAttr(".shrs");
	float shaftShrink = feather->m_shaftShrink;
	writeFloatAttr(".shrs", &shaftShrink);
	
	if(!hasNamedAttr(".bws")) addFloatAttr(".bws");
	float barbWidthScale = feather->m_barbWidthScale;
	writeFloatAttr(".bws", &barbWidthScale);
	
	return 1;
}

char HFeather::load(MlFeather * feather)
{
	if(!hasNamedAttr(".featherId"))
		return 0;
	
	int id = 0;
	readIntAttr(".featherId", &id);
	
	feather->setFeatherId(id);
	
	int typ = 0;
	if(hasNamedAttr(".typ")) readIntAttr(".typ", &typ);
	feather->setType(typ);
	
	int nseg = 4;
	if(!hasNamedAttr(".nseg"))
		return 0;
		
	readIntAttr(".nseg", &nseg);
	
	feather->createNumSegment(nseg);
	
	if(!hasNamedData(".quill"))
		return 0;
		
	readFloatData(".quill", nseg, feather->quilly());
	
	int nvv = feather->numVaneVertices();
	if(!hasNamedAttr(".nvv"))
		return 0;
		
	readIntAttr(".nvv", &nvv);
	
	if(!hasNamedData(".vv"))
		return 0;
		
	readFloatData(".vv", nvv * 2, (float *)feather->uvDisplace());
	
	Vector2F uv(4.f, 4.f);
	
	if(hasNamedAttr(".uv"))
		readFloatAttr(".uv", (float *)(&uv));
		
	feather->setBaseUV(uv);
	
	int rs = 10;
	if(hasNamedAttr(".rs")) readIntAttr(".rs", &rs);
	feather->setResShaft(rs);
	
	int rb = 9;
	if(hasNamedAttr(".rb")) readIntAttr(".rb", &rb);
	feather->setResBarb(rb);
	
	int nsep = 2;
	if(hasNamedAttr(".nsep")) readIntAttr(".nsep", &nsep);
	feather->setNumSeparate(nsep);
	
	float kfuzz = 0.f;
	if(hasNamedAttr(".kfuzz")) readFloatAttr(".kfuzz", &kfuzz);
	feather->setFuzzy(kfuzz);
	
	float ksep = 0.f;
	if(hasNamedAttr(".ksep")) readFloatAttr(".ksep", &ksep);
	feather->setSeparateStrength(ksep);
	
	float barbShrink = 0.5f;
	if(hasNamedAttr(".shrb")) readFloatAttr(".shrb", &barbShrink);
	feather->m_barbShrink = barbShrink;
	
	float shaftShrink = 0.5f;
	if(hasNamedAttr(".shrs")) readFloatAttr(".shrs", &shaftShrink);
	feather->m_shaftShrink = shaftShrink;
	
	float barbWidthScale = 0.67f;
	if(hasNamedAttr(".bws")) readFloatAttr(".bws", &barbWidthScale);
	feather->m_barbWidthScale = barbWidthScale;
		
	feather->computeLength();
	feather->computeTexcoord();
	feather->verbose();
	
	return 1;
}

int HFeather::loadId()
{
	if(!hasNamedAttr(".featherId"))
		return 0;
	
	int id = 0;
	readIntAttr(".featherId", &id);
	
	return id;
}
