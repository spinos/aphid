/*
 *  HDoc.cpp
 *  helloHdf
 *
 *  Created by jian zhang on 6/12/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HDocument.h"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "hdf5.h"
using namespace boost::filesystem;
char HDocument::open(const char * filename, OpenMode mode)
{
	char fileExists = 0;
	path file_path(filename);
	
	if ( is_regular_file( file_path ) ) {
		 fileExists = 1;
	}
	
	if(mode == oReadOnly) {
		if(!fileExists) {
			fCurrentError = eFileAccessFailure;
			return 0;
		}

		fFileId = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
		if(fFileId < 0) {
			fCurrentError = eFileAccessFailure;
			return 0;
		}
	}
	
	if(mode == oCreate) {
		fFileId = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if(fFileId < 0) {
			fCurrentError = eFileAccessFailure;
			return 0;
		}
	}

	if(mode == oReadAndWrite) {
		if(fileExists)
			fFileId = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
		else
			fFileId = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		
		if(fFileId < 0) {
			fCurrentError = eFileAccessFailure;
			return 0;
		}
	}
	fFileName = filename;
	fCurrentError = eNoError;
	return 1;
}

char HDocument::close()
{
	if(!isOpened())
		return 1;
		
	if(H5Fclose(fFileId) < 0) {
		fCurrentError = eCloseFailure;
		return 0;
	}
	fFileName = "";
	fFileId = -1;
	return 1;
}

char HDocument::isOpened() const
{
	return fFileId > 0;
}

char HDocument::checkExist(const std::string & path)
{
	if(!H5Lexists(fFileId, path.c_str(), H5P_DEFAULT))
		return 0;
		
	return 1;
}

char HDocument::deleteObject(const std::string & path, hid_t parentId)
{
    if(parentId == 0) parentId = fFileId;
	if(H5Ldelete(parentId, path.c_str(), H5P_DEFAULT) < 0) {
		fCurrentError = eDeleteFailure;
		return 0;
	}
	return 1;
}

std::string HDocument::fileName() const
{
    return fFileName;
}

