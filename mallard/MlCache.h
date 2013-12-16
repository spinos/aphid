/*
 *  MlCache.h
 *  mallard
 *
 *  Created by jian zhang on 10/12/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <CacheFile.h>

class MlCache : public CacheFile {
public:
	MlCache();
	virtual ~MlCache();
	
	virtual bool doRead(const std::string & name);
	virtual bool doCopy(const std::string & name);
	
	void setSceneName(const std::string & name);
	std::string sceneName();
	
	bool isBaked(unsigned n) const;
private:
	void setCachedSlices(const std::string & name);
	std::string m_sceneName;
};