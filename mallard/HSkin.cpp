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
HSkin::HSkin(const std::string & path) : HBase(path) {}
	
char HSkin::save(MlSkin * s)
{
	int nc = s->numFeathers();
	
	if(!hasNamedAttr(".nc"))
		addIntAttr(".nc");
		
	writeIntAttr(".nc", &nc);
	std::cout<<" num feather instances "<<nc<<"\n";
	
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
		std::cout<<"no nc";
		return 0;
	}
	
	int nc = 0;
	readIntAttr(".nc", &nc);
	std::cout<<" num feather instances "<<nc<<"\n";
	
	const unsigned fullSize = nc * sizeof(MlCalamus);
	std::cout<<" full size "<<fullSize<<"\n";
	
	s->setNumFeathers(nc);
	
	MlCalamusArray * arr = s->getCalamusArray();
	
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
		
		readCharData(".cs", fullSize, d, &p);
		
		blockStart += arr->numElementPerBlock() * sizeof(MlCalamus);
	}
	
	arr->setIndex(nc);
	return 1;
}