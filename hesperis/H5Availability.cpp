/*
 *  H5Availability.cpp
 *  opium
 *
 *  Created by jian zhang on 6/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <SHelper.h>
#include <HObject.h>
#include <HIntAttribute.h>
#include <sstream>
#include "H5Availability.h"
#include <HBase.h>
#include <AHelper.h>

H5Availability::H5Availability() {}
H5Availability::~H5Availability() {}

char  H5Availability::openFile(const char* filename, HDocument::OpenMode accessFlag)
{
	if(fFileStatus.count(filename) < 1) {
        HDocument * d = new HDocument;
		d->open(h5FileName(filename).c_str(), accessFlag);
        if(d->isOpened()) {
            // m_samplers[filename] = SampleFrame();
            fFileStatus[filename] = d;       
        }
        else {
            AHelper::Info<const char *>("h5 not opened", filename);
            return 0;
        }
	}
	
	HObject::FileIO = *fFileStatus[filename];
    
	return 1;
}

char H5Availability::closeFile(const char* filename)
{
	if(fFileStatus.count(filename) < 1)
		return 1;
		
	if(!fFileStatus[filename]->isOpened())
		return 1;
		
	return fFileStatus[filename]->close();
}

std::string H5Availability::h5FileName(const char* filename)
{
	std::string res(filename);
	SHelper::changeFilenameExtension(res, "h5");
	return res;
}

std::string H5Availability::closeAll()
{
    std::stringstream sst;
	std::map<std::string, HDocument *>::iterator it = fFileStatus.begin();
	for(; it != fFileStatus.end(); ++it) {
		HDocument * doc = it->second;
		sst<<" "<<it->first<<"\n";
		if(doc->isOpened()) doc->close();
        delete doc;
	}
    fFileStatus.clear();
	return sst.str();
}
//~: