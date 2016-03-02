/*
 *  Holder.cpp
 *  helloHdf
 *
 *  Created by jian zhang on 10/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "Holder.h"
#include <HBase.h>
#include <boost/timer.hpp>
#include <iostream>
using namespace aphid;

Holder::Holder() {}
Holder::~Holder() {}

void Holder::Run(const char * filename) 
{
	if(!HObject::FileIO.open(filename, HDocument::oReadAndWrite)) 
		std::cout << "\n warning cannot open file  " << filename <<"\n"; 
		
	HBase w("/");
	
	
	boost::timer met;
	met.restart();
	for(;;) { if(met.elapsed() > 10) break; }
	
	std::cout<<"\n / has n children "<<w.numChildren();
	w.close();
	
	std::cout << "\n released aft 10 seconds \n";
}