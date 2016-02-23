/*
 *  MultiPlaybackFile.cpp
 *  opium
 *
 *  Created by jian zhang on 6/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <SHelper.h>
#include <HObject.h>
#include <HIntAttribute.h>
#include "MultiPlaybackFile.h"
#include "APlaybackFile.h"

namespace aphid {

MultiPlaybackFile::MultiPlaybackFile() {}
MultiPlaybackFile::~MultiPlaybackFile() {}

bool MultiPlaybackFile::addFile(APlaybackFile * file)
{
	if(!file->isOpened()) {
		std::cout<<"\n playback file is not opened "<<file->fileName();
		return false;
	}
	
	fFileStatus[file->fileName()] = file; 
	
	return true;
}

bool MultiPlaybackFile::closeFile(const std::string & fileName)
{
	APlaybackFile * f = namedFile(fileName);
	if(!f) {
		std::cout<<"\n playback file is not opened "<<fileName;
		return false;
	}
	return f->close();
}

void MultiPlaybackFile::cleanup()
{
	HObject::FileIO.close();
	std::map<std::string, APlaybackFile *>::iterator iter = fFileStatus.begin();
	for(; iter != fFileStatus.end(); ++iter) {
		iter->second->close();
	}
    fFileStatus.clear();
}

APlaybackFile * MultiPlaybackFile::namedFile(const std::string & fileName) const
{
	std::map<std::string, APlaybackFile *>::const_iterator it = fFileStatus.find(fileName);
	if(it == fFileStatus.end()) return 0;
	return it->second;
}

}
//~: