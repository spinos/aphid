/*
 *  NTreeIO.h
 *  
 *
 *  Created by jian zhang on 3/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HDocument.h>
#include <ConvexShape.h>

namespace aphid {

class NTreeIO {

public:
	NTreeIO();
	
	bool begin(const std::string & filename, 
				HDocument::OpenMode om = HDocument::oReadOnly);
	void end();
	
	bool findGrid(std::string & name,
				const std::string & grpName="/");
	
	bool findTree(std::string & name,
				const std::string & grpName="/");
	
	cvx::ShapeType gridValueType(const std::string & name);
	
	bool loadSphereTree(const std::string & name);
	
protected:

private:

};

}