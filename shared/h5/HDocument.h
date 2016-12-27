#ifndef APH_H_DOCUMENT_H
#define APH_H_DOCUMENT_H

/*
 *  HDoc.h
 *  helloHdf
 *
 *  Created by jian zhang on 6/12/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <hdf5.h>

#include <string>

namespace aphid {

class HDocument {
public:
	enum OpenMode {
		oCreate,
		oReadOnly,
		oReadAndWrite
	};
	
	enum ErrorMessage {
		eNoError,
		eFileAccessFailure,
		eInvalidPath,
		eCloseFailure,
		eDeleteFailure,
		eDataSpaceDimensionMissMatch,
		eLackOfDataSpaceDimension
	};
	
	HDocument() : fFileId(-1) {}
	~HDocument() {}
	
	char open(const char * filename, OpenMode mode);
	char close();
	char isOpened() const;
	
	char checkExist(const std::string & path);

	char deleteObject(const std::string & path, hid_t parentId = 0);
	
	std::string fileName() const;
	
	std::string fFileName;
	hid_t fFileId;
	ErrorMessage fCurrentError;
};

}
#endif        //  #ifndef HDOCUMENT_H
