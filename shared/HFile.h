/*
 *  HFile.h
 *  mallard
 *
 *  Created by jian zhang on 10/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HDocument.h>
#include <BaseFile.h>

class HFile : public BaseFile {
public:
	HFile();
	HFile(const char * name);
	
	virtual bool close();
	
protected:
	void useDocument();
	void setDocument(const HDocument & doc);
private:
	HDocument m_doc;
};