/*
 *  HSkin.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HSkin.h"
#include <MlSkin.h>
#include <MlCalamusArray.h>
#define CURRENT_API_VERTION 1
HSkin::HSkin(const std::string & path) : HBase(path) {}
	
char HSkin::save(MlSkin * s)
{
	int apiVer = CURRENT_API_VERTION;
	if(!hasNamedAttr(".apiver")) addIntAttr(".apiver");
	writeIntAttr(".apiver", &apiVer);
	std::cout<<" API version: "<<apiVer<<"\n";
	
	int nc = s->numFeathers();	
	if(!hasNamedAttr(".nc")) addIntAttr(".nc");
	writeIntAttr(".nc", &nc);
	std::cout<<" num feather instances: "<<nc<<"\n";
	
	if(nc < 1) return 0;
	
	const unsigned fullSize = nc * sizeof(MlCalamus);
	std::cout<<" full size "<<fullSize<<"\n";
	
	if(!hasNamedData(".cs"))
		addCharData(".cs", fullSize);
		
	MlCalamusArray * arr = s->getCalamusArray();
	std::cout<<" elm per blk "<<arr->numElementPerBlock()<<"\n";
	
	int blockStart = 0;
	for(unsigned i = 0; i < arr->numBlocks(); i++) {
		unsigned blockSize = arr->numElementsInBlock(i, nc);
		if(blockSize == 0) break;
		
		std::cout<<" elm in blk "<<i<<" "<<blockSize<<"\n";
		
		char * d = arr->getBlock(i);
		HDataset::SelectPart p;
		p.start[0] = blockStart;
		p.count[0] = 1;
		p.block[0] = blockSize * sizeof(MlCalamus);
		
		std::cout<<" block "<<i<<" ["<<p.start[0]<<", "<<p.start[0] + p.block[0]<<"]\n";
		
		writeCharData(".cs", fullSize, d, &p);
		
		blockStart += arr->numElementPerBlock() * sizeof(MlCalamus);
	}
	
	return 1;
}

char HSkin::load(MlSkin * s)
{
	if(!hasNamedAttr(".nc")) {
		std::cout<<"ERROR: invalid data has no .nc!\n";
		return 0;
	}
	
	int nc = 0;
	readIntAttr(".nc", &nc);
	std::cout<<" num feather instances: "<<nc<<"\n";
	
	int apiVer = 0;
	if(hasNamedAttr(".apiver")) readIntAttr(".apiver", &apiVer);
	std::cout<<" API version: "<<apiVer<<"\n";
	
	s->setNumFeathers(nc);
	
	MlCalamusArray * arr = s->getCalamusArray();
	
	if(apiVer == CURRENT_API_VERTION) readCurData(arr, nc);
	else if(apiVer == 0) readV0Data(arr, nc);
	
	arr->setIndex(nc);
	return 1;
}

void HSkin::readCurData(MlCalamusArray * arr, const unsigned & num)
{
	const unsigned fullSize = num * sizeof(MlCalamus);
	std::cout<<" data size "<<fullSize<<"\n";
	
	int blockStart = 0;
	for(unsigned i = 0; i < arr->numBlocks(); i++) {
		const unsigned blockSize = arr->numElementsInBlock(i, num);
		if(blockSize == 0) break;
		
		std::cout<<" elm in blk "<<i<<" "<<blockSize<<"\n";
		
		char * d = arr->getBlock(i);
		
		HDataset::SelectPart p;
		p.start[0] = blockStart;
		p.count[0] = 1;
		p.block[0] = blockSize * sizeof(MlCalamus);
		
		std::cout<<" block "<<i<<" ["<<p.start[0]<<", "<<p.start[0] + p.block[0]<<"]\n";
		
		readCharData(".cs", fullSize, d, &p);
		
		blockStart += arr->numElementPerBlock() * sizeof(MlCalamus);
	}
}

void HSkin::readV0Data(MlCalamusArray * arr, const unsigned & num)
{
	boost::scoped_array<CalaVer0> surrogate(new CalaVer0[524288/sizeof(CalaVer0)]);
	const unsigned fullSize = num * sizeof(MlCalamus);
	std::cout<<" data size "<<fullSize<<"\n";
	
	int blockStart = 0;
	for(unsigned i = 0; i < arr->numBlocks(); i++) {
		const unsigned blockSize = arr->numElementsInBlock(i, num);
		if(blockSize == 0) break;
		
		std::cout<<" elm in blk "<<i<<" "<<blockSize<<"\n";
		
		char * d = arr->getBlock(i);
		
		HDataset::SelectPart p;
		p.start[0] = blockStart;
		p.count[0] = 1;
		p.block[0] = blockSize * sizeof(MlCalamus);
		
		std::cout<<" block "<<i<<" ["<<p.start[0]<<", "<<p.start[0] + p.block[0]<<"]\n";
		
		readCharData(".cs", fullSize, d, &p);
		readCharData(".cs", fullSize, (char *)surrogate.get(), &p);
		
		const unsigned nc = blockSize;
		MlCalamus * c = (MlCalamus *)d;
		for(unsigned i = 0; i < nc; i++) {
			c[i].setLength(surrogate[i].m_scale);
			c[i].setRotateX(surrogate[i].m_rotX);
			c[i].setCurlAngle(surrogate[i].m_rotY);
		}
		
		blockStart += arr->numElementPerBlock() * sizeof(MlCalamus);
	}
}
