/*
 *  HOption.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "HOption.h"

#include <AllHdf.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <RenderOptions.h>
HOption::HOption(const std::string & path) : HBase(path) 
{
}

HOption::~HOption() {}

char HOption::save(RenderOptions * opt)
{
	
	int aas = opt->AASample();
	if(!hasNamedAttr(".aas"))
		addIntAttr(".aas");
	
	writeIntAttr(".aas", &aas);
	
	int rw = opt->renderImageWidth();
	if(!hasNamedAttr(".rw"))
		addIntAttr(".rw");
	
	writeIntAttr(".rw", &rw);
		
	int rh = opt->renderImageHeight();
	if(!hasNamedAttr(".rh"))
		addIntAttr(".rh");
	
	writeIntAttr(".rh", &rh);
	
	int msd = opt->maxSubdiv();
	if(!hasNamedAttr(".msd"))
		addIntAttr(".msd");
	
	writeIntAttr(".msd", &msd);
	
	int uds = 0;
	if(opt->useDisplaySize()) uds = 1;
	
	if(!hasNamedAttr(".uds"))
		addIntAttr(".uds");
	
	writeIntAttr(".uds", &uds);

	return 1;
}

char HOption::load(RenderOptions * opt)
{
	int aas = 4;
	if(hasNamedAttr(".aas"))
		readIntAttr(".aas", &aas);
	opt->setAASample(aas);
	
	int rw = 400;
	if(hasNamedAttr(".rw"))
		readIntAttr(".rw", &rw);
	opt->setRenderImageWidth(rw);
		
	int rh = 300;
	if(hasNamedAttr(".rh"))
		readIntAttr(".rh", &rh);
	opt->setRenderImageHeight(rh);
	
	int msd = 3;
	if(hasNamedAttr(".msd"))
		readIntAttr(".msd", &msd);
	opt->setMaxSubdiv(msd);
	
	int uds = 0;
	if(hasNamedAttr(".uds"))
		readIntAttr(".uds", &uds);
		
	if(uds == 1) opt->setUseDisplaySize(true);
	else opt->setUseDisplaySize(false);
	
	return 1;
}