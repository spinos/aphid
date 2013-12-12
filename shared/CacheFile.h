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
#include <BoundingBox.h>
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
	bool openSliceFloat(const std::string & entryName, const std::string & sliceName);
	bool openSliceVector3(const std::string & entryName, const std::string & sliceName);
	void closeSlice(const std::string & entryName, const std::string & sliceName);
	
	void saveEntrySize(const std::string & entryName, unsigned size);
	unsigned entrySize(const std::string & entryName);
	
	void writeSliceVector3(const std::string & entryName, const std::string & sliceName, unsigned start, unsigned count, Vector3F * data);
	void readSliceVector3(const std::string & entryName, const std::string & sliceName, unsigned start, unsigned count, Vector3F * data);

	void writeSliceFloat(const std::string & entryName, const std::string & sliceName, unsigned start, unsigned count, Vector3F * data);
	void readSliceFloat(const std::string & entryName, const std::string & sliceName, unsigned start, unsigned count, Vector3F * data);

	void setCached(const std::string & entryName, const std::string & sliceName, unsigned size);
	unsigned isCached(const std::string & entryName, const std::string & sliceName);
	void clearCached();
	
	unsigned numCachedSlices(const std::string & entryName) const;
	unsigned cacheSliceNames(const std::string & entryName, std::vector<std::string> & dst) const;

	void setBounding(const std::string & name, const BoundingBox & box);
	void getBounding(const std::string & name, BoundingBox & box);
	
	void setTranslation(const std::string & name, const Vector3F & at);
	void getTranslation(const std::string & name, Vector3F & at);
private:
	std::map<std::string, HBase *> m_entries;
	std::map<std::string, HDataset *> m_slices;
	std::map<std::string, unsigned> m_cachedSlices;
};

