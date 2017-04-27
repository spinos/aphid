/*
 *  ExportExample.cpp
 *  
 *
 *  Created by jian zhang on 4/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ExportExample.h"
#include "Vegetation.h"
#include <h5/HObject.h>
#include <h5/HDocument.h>
#include <HGardenExample.h>
#include <boost/format.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace boost::posix_time;
using namespace aphid;

ExportExample::ExportExample(Vegetation * vege) :
m_vege(vege)
{}

void ExportExample::exportToFile(const std::string & filename)
{
	HDocument h5doc;
	if(!h5doc.open(filename.c_str(), HDocument::oCreate)) {
		std::cout<<"ExportExample error cannot open gde file "<<filename;
        return;
	}
	
	HObject::FileIO = h5doc;
	
	const ptime tnow = second_clock::local_time();
	const std::string empname = str(boost::format("/gde%1%") % to_iso_string(tnow));
	
	HGardenExample wld(empname);
	wld.save(m_vege);
	wld.close();
	
	h5doc.close();
}
