/*
 *  EasemodelUtil.h
 *  hc
 *
 *  Created by jian zhang on 4/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */	
#pragma once

#include <BaseMesh.h>
#include <PatchMesh.h>
class EasyModelIn;
namespace ESMUtil
{
void Import(const char * filename, BaseMesh * dst);
void ImportPatch(const char * filename, PatchMesh * dst);

void Export(const char * filename, BaseMesh * src);

void baseImport(EasyModelIn *esm, BaseMesh * dst);
}