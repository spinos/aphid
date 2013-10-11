/*
 *  CacheFile.h
 *  mallard
 *
 *  Created by jian zhang on 10/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
 
#pragma once
#include <HFile.h>
#include <BaseState.h>
#include <map>
#include <AllMath.h>
class HBase;
class HDataset;
class CacheFile : public HFile, public BaseState {
public:
    CacheFile();
	CacheFile(const char * name);
    
	virtual void doClose();
	
	HBase * getNamedEntry(const std::string & name);
	
	void openEntry(const std::string & name);
	void closeEntry(const std::string & name);
	bool openSlice(const std::string & entryName, const std::string & sliceName);
	void closeSlice(const std::string & entryName, const std::string & sliceName);
	void saveEntrySize(const std::string & entryName, unsigned size);
	
	void writeSliceVector3(const std::string & entryName, const std::string & sliceName, unsigned start, unsigned count, Vector3F * data);
	void readSliceVector3(const std::string & entryName, const std::string & sliceName, unsigned start, unsigned count, Vector3F * data);

	void setCached(const std::string & entryName, const std::string & sliceName, unsigned size);
	unsigned isCached(const std::string & entryName, const std::string & sliceName);
	void clearCached();
	
	unsigned numCachedSlices(const std::string & entryName) const;

private:
	std::map<std::string, HBase *> m_entries;
	std::map<std::string, HDataset *> m_slices;
	std::map<std::string, unsigned> m_cachedSlices;
};

