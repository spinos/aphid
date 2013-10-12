/*
 *  MlCache.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/12/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlCache.h"
#include <AllHdf.h>
#include <HBase.h>
MlCache::MlCache() {}
MlCache::~MlCache() {}

bool MlCache::doCopy(const std::string & name)
{
	std::cout<<"copy cache to "<<name;

	useDocument();
	openEntry("/p");
	int psize = entrySize("/p");

	if(psize < 1) return false;
	
	std::vector<std::string> sliceNames;
	unsigned numSlices = cacheSliceNames("/p", sliceNames);
	std::cout<<" n slice "<<numSlices<<"in /p\n";
	if(numSlices < 2) return false;
	
	MlCache tgt;
	if(!tgt.create(name)) return false;
	
	tgt.useDocument();
	tgt.openEntry("/info");
	HBase *info = tgt.getNamedEntry("/info");
	info->addStringAttr(".scene", m_sceneName.size());
	info->writeStringAttr(".scene", m_sceneName);
	tgt.closeEntry("/info");
	tgt.openEntry("/p");
	tgt.saveEntrySize("/p", psize);
	
	const unsigned blockL = 16384;
	Vector3F * b = new Vector3F[blockL];
	unsigned i, j, start, count;
	for(i = 0; i < sliceNames.size(); i++) {
		std::string aslice = HObject::PartialPath("/p", sliceNames[i]);
		
		useDocument();
		openSlice("/p", aslice);
		tgt.useDocument();
		tgt.openSlice("/p", aslice);
		start = 0;
		count = blockL;
		for(j = 0; j <= psize / blockL; j++) {
			start = blockL * j;
			if(j == psize / blockL)
				count = psize - start;
				
				useDocument();
				readSliceVector3("/p", aslice, start, count, b);
				
				tgt.useDocument();
				tgt.writeSliceVector3("/p", aslice, start, count, b);
				
		}
		useDocument();
		closeSlice("/p", aslice);
		tgt.useDocument();
		tgt.closeSlice("/p", aslice);
	}
	delete[] b;
	useDocument();
	closeEntry("/p");
	tgt.useDocument();
	tgt.closeEntry("/p");
	tgt.close();
	return true;
}

void MlCache::setSceneName(const std::string & name)
{
	m_sceneName = name;
}