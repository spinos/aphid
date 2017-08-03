/*
 *  ExportExample.h
 *  garden
 *
 *  Created by jian zhang on 4/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_EXPORT_EXAMPLE_H
#define GAR_EXPORT_EXAMPLE_H

#include <string>

class Vegetation;

class ExportExample {

	Vegetation * m_vege;
	
public:
	ExportExample(Vegetation * vege);
	
	void exportToFile(const std::string & filename);
	
protected:

private:
};

#endif