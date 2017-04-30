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
	if(info->hasNamedAttr(".range"))
	    info->readIntAttr(".range", m_bakeRange);
	closeEntry("/info");
	
	setCachedSlices("/p");
	setCachedSlices("/tang");
	setCachedSlices("/ang");
	return true;
}

bool MlCache::doCopy(const std::string & name)
{
	std::cout<<"copy cache to "<<name;

	useDocument();
	openEntry("/ang");
	int nang = entrySize("/ang");

	if(nang < 1) return false;
	
	std::vector<std::string> sliceNames;
	unsigned numSlices = cacheSliceNames("/ang", sliceNames);
	std::cout<<" n slice "<<numSlices<<"in /ang\n";
	if(numSlices < 2) return false;
	
	openEntry("/tang");
	int tsize = entrySize("/tang");
	openEntry("/p");
	int ptsize = entrySize("/p");
	
	MlCache tgt;
	if(!tgt.create(name)) return false;
	
	tgt.useDocument();
	tgt.openEntry("/info");
	HBase *info = tgt.getNamedEntry("/info");
	info->addStringAttr(".scene", m_sceneName.size());
	info->writeStringAttr(".scene", m_sceneName);
	info->addIntAttr(".range", 2);
	info->writeIntAttr(".range", m_bakeRange);
	tgt.closeEntry("/info");
	
	tgt.openEntry("/ang");
	tgt.saveEntrySize("/ang", nang);
	
	tgt.openEntry("/tang");
	tgt.saveEntrySize("/tang", tsize);
	
	tgt.openEntry("/p");
	tgt.saveEntrySize("/p", ptsize);
	
	const unsigned blockL = 4096;
	float * b = new float[blockL];
	Vector3F * bp = new Vector3F[blockL];
	Matrix33F * bm = new Matrix33F[blockL];
	
	BoundingBox box;
	Vector3F center;
	unsigned i, j, start, count;
	for(i = 0; i < sliceNames.size(); i++) {
		std::string aslice = HObject::PartialPath("/ang", sliceNames[i]);
		if(aslice == "-9999") continue;
		
		useDocument();
		openSliceFloat("/ang", aslice);
		openSliceVector3("/p", aslice);
		openSliceMatrix33("/tang", aslice);
		
		tgt.useDocument();
		tgt.openSliceFloat("/ang", aslice);
		tgt.openSliceVector3("/p", aslice);
		tgt.openSliceMatrix33("/tang", aslice);
		
		start = 0;
		count = blockL;
		for(j = 0; j <= nang/blockL; j++) {
			if(j== nang/blockL)
				count = nang%blockL;
				
			start = j * blockL;

			useDocument();
			readSliceFloat("/ang", aslice, start, count, b);
		
			tgt.useDocument();
			tgt.writeSliceFloat("/ang", aslice, start, count, b);
		}
		
		start = 0;
		count = blockL;
		for(j = 0; j <= tsize/blockL; j++) {
			if(j== tsize/blockL)
				count = tsize%blockL;
				
			start = j * blockL;	
			useDocument();
			readSliceVector3("/p", aslice, start, count, bp);
			readSliceMatrix33("/tang", aslice, start, count, bm);
			
			tgt.useDocument();
			tgt.writeSliceVector3("/p", aslice, start, count, bp);
			tgt.writeSliceMatrix33("/tang", aslice, start, count, bm);

		}
		
		useDocument();
		closeSlice("/ang", aslice);
		closeSlice("/tang", aslice);
		closeSlice("/p", aslice);
		tgt.useDocument();
		tgt.closeSlice("/ang", aslice);
		tgt.closeSlice("/tang", aslice);
		tgt.closeSlice("/p", aslice);
		
		getBounding(aslice, box);
		getTranslation(aslice, center);
		
		tgt.setBounding(aslice, box);
		tgt.setTranslation(aslice, center);
		tgt.flush();
	}
	delete[] b;
	delete[] bp;
	delete[] bm;
	
	useDocument();
	closeEntry("/ang");
	closeEntry("/tang");
	closeEntry("/p");
	
	tgt.useDocument();
	tgt.closeEntry("/ang");
	tgt.closeEntry("/tang");
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

bool MlCache::isBaked(unsigned n) const
{
	return numCachedSlices("/ang") == n;
}

void MlCache::setCachedSlices(const std::string & name)
{
	openEntry(name);
	HBase * p = getNamedEntry(name);
	int nf = p->numChildren();
	unsigned csize = entrySize(name);
	for(int i = 0; i < nf; i++) {
		if(p->isChildData(i))
			setCached(name, p->getChildName(i), csize);
	}
	
	closeEntry(name);
}

void MlCache::setBakeRange(int low, int high)
{
    m_bakeRange[0] = low;
    m_bakeRange[1] = high;
}
	
void MlCache::bakeRange(int & low, int & high) const
{
    low = m_bakeRange[0];
    high = m_bakeRange[1];
}