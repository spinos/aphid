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

MlCache::MlCache() 
{
    m_sceneName = "unknown";
}

MlCache::~MlCache() {}

bool MlCache::doRead(const std::string & name)
{
	if(!HFile::doRead(name)) return false;
	
	openEntry("/info");
	HBase *info = getNamedEntry("/info");
	if(info->hasNamedAttr(".scene"))
	    info->readStringAttr(".scene", m_sceneName);
	closeEntry("/info");
	
	openEntry("/p");
	HBase * p = getNamedEntry("/p");
	int nf = p->numChildren();
	unsigned csize = entrySize("/p");
	for(int i = 0; i < nf; i++) {
		if(p->isChildData(i))
			setCached("/p", p->getChildName(i), csize);
	}
	
	closeEntry("/p");
	return true;
}

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
	BoundingBox box;
	Vector3F center;
	unsigned i, j, start, count;
	for(i = 0; i < sliceNames.size(); i++) {
		std::string aslice = HObject::PartialPath("/p", sliceNames[i]);
		if(aslice == "-9999") continue;
		useDocument();
		openSliceVector3("/p", aslice);
		tgt.useDocument();
		tgt.openSliceVector3("/p", aslice);
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
		
		getBounding(aslice, box);
		getTranslation(aslice, center);
		
		tgt.setBounding(aslice, box);
		tgt.setTranslation(aslice, center);
		tgt.flush();
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

std::string MlCache::sceneName()
{
    return m_sceneName;
}