/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "Parameter.h"
#include "JuliaWorld.h"
#include "QuatJulia.h"
#include "JuliaTree.h"

int main(int argc, char *argv[])
{
	jul::Parameter param(argc, argv);
	if(!param.isValid() || param.operation() == jul::Parameter::kHelp ) {
		jul::Parameter::PrintHelp();
		return 1;
	}
	
	if(param.operation() == jul::Parameter::kGenerate ) {
		std::cout<<"\n generate julia set ";
		jul::QuatJulia julia(&param);
	}
	
	if(param.operation() == jul::Parameter::kBuildTree ) {
		std::cout<<"\n build kdntree ";
		jul::JuliaTree t(&param);
	}
	
	jul::JuliaWorld wd;
	if(param.operation() == jul::Parameter::kInitialize ) {
		wd.create(param);
	}
	
	if(param.operation() == jul::Parameter::kInsert ) {
		wd.insert(param);
	}
	return 0;
}
