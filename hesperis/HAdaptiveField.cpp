/*
 *  HAdaptiveField.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/5/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HAdaptiveField.h"
#include "AdaptiveField.h"
#include "BaseBuffer.h"
#include <MortonHash.h>

struct IOCellHash {
	unsigned code;
	int level;
	int visited;
	unsigned index;
};

HAdaptiveField::HAdaptiveField(const std::string & path)
: HField(path) {}
HAdaptiveField::~HAdaptiveField() {}
	
char HAdaptiveField::verifyType()
{
	if(!hasNamedAttr(".ncells"))
        return 0;
		
	if(!hasNamedAttr(".origin_span"))
        return 0;
    
    if(!hasNamedAttr(".max_level"))
        return 0;
		
	return HField::verifyType();
}

char HAdaptiveField::save(AdaptiveField * fld)
{
    int nc = fld->numCells();
    std::cout<<"\n hadaptivefield save n cells "<<nc;
	
	if(!hasNamedAttr(".ncells"))
		addIntAttr(".ncells");
	
	writeIntAttr(".ncells", &nc);
	
	if(!hasNamedAttr(".origin_span"))
        addFloatAttr(".origin_span", 4);
		
	float originSpan[4];
	originSpan[0] = fld->origin().x;
	originSpan[1] = fld->origin().y;
	originSpan[2] = fld->origin().z;
	originSpan[3] = fld->span();
	writeFloatAttr(".origin_span", originSpan);
    
    if(!hasNamedAttr(".max_level"))
        addIntAttr(".max_level");
    
    int ml = fld->maxLevel();
    writeIntAttr(".ncells", &ml);
	
	BaseBuffer dhash;
	dhash.create(nc * 16);
	IOCellHash * dst = (IOCellHash *)dhash.data();
	
	sdb::CellHash * c = fld->cells();
	c->begin();
	while(!c->end()) {
		dst->code = c->key();
		dst->level = c->value()->level;
		dst->visited = c->value()->visited;
		dst->index = c->value()->index;
		dst++;
		c->next();
	}
	
	if(!hasNamedData(".cellHash"))
		addCharData(".cellHash", nc*16);
		
	writeCharData(".cellHash", nc*16, dhash.data());
		
	return HField::save(fld);
}

char HAdaptiveField::load(AdaptiveField * fld)
{
	int nc = 1;
	readIntAttr(".ncells", &nc);
	
	float originSpan[4];
	readFloatAttr(".origin_span", originSpan);
    
    int ml = 7;
    readIntAttr(".ncells", &ml);
	
	fld = new AdaptiveField(originSpan);
    fld->setMaxLevel(ml);
	
	BaseBuffer dhash;
	dhash.create(nc * 16);
	readCharData(".cellHash", nc*16, dhash.data());
	
	IOCellHash * src = (IOCellHash *)dhash.data();
	
	unsigned i = 0;
	for(;i<nc;i++) {
		fld->addCell(src->code, src->level, src->visited, src->index);
		src++;
	}
	
	return HField::load(fld);
}