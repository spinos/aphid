/*
 *  MultiPlaybackFile.h
 *  opium
 *
 *  Created by jian zhang on 6/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <string>
#include <map>

namespace aphid {

class APlaybackFile;
class MultiPlaybackFile {
public:
	MultiPlaybackFile();
	~MultiPlaybackFile();
	
	bool addFile(APlaybackFile * file);
	APlaybackFile * namedFile(const std::string & fileName) const;
	bool closeFile(const std::string & filename);
	void cleanup();
	
protected:
	
private:
	std::map<std::string, APlaybackFile *> fFileStatus;
};

}