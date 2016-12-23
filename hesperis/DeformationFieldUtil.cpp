/*
 *  GeometryUtil.cpp
 *  opium
 *
 *  Created by jian zhang on 3/3/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "DeformationFieldUtil.h"
#include "animUtil.h"
#include "transformUtil.h"
#include "animIO.h"
#include "sceneIO.h"
#include <sstream>
#include <string>

#include <foundation/SHelper.h>
#include <ASearchHelper.h>
#include <AAttributeHelper.h>
#include "AttribUtil.h"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <boost/format.hpp>

#include <HesperisFile.h>
#include <HesperisAttributeIO.h>
#include <H5IO.h>

using namespace boost::filesystem;
using namespace std;

DeformationFieldUtil::DeformationFieldUtil()
{}

void DeformationFieldUtil::dump(const char *filename, 
                            MDagPathArray &active_list)
{
    if(active_list.length() < 1) {
        MGlobal::displayInfo("insufficient selection! select group(s) of geometries to push deformation field.");
        return;
    }
	
	if(AFrameRange::isValid() ) {
	    AHelper::Info<int>("DeformationFieldUtil rewind to frame ", 
	                            BaseUtil::FirstFrame);
        MGlobal::executeCommand( MString("currentTime ")
                                +BaseUtil::FirstFrame);
    }
	
	MDagPathArray tms;
	getActiveTransforms(tms, active_list);
	
	SceneIO doc;
	doc.create(filename);
	doc.recordTime();
	if(AFrameRange::isValid()) {
		doc.addFraneRange(FirstFrame, LastFrame);
		doc.addSPF(SamplesPerFrame);
	}
    saveFormatVersion(doc, 3.f);
	AnimUtil::ResolveFPS(HesperisAnimIO::SecondsPerFrame);
    doc.save(filename);
	doc.free();
	
	
    
}

//:~
