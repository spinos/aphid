/*
 *  HNTree.cpp
 *  
 *
 *  Created by jian zhang on 3/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "HNTree.h"

namespace aphid {

HBaseNTree::HBaseNTree(const std::string & name) :
HBase(name) {}

HBaseNTree::~HBaseNTree() {}
	
char HBaseNTree::verifyType()
{
	if(!hasNamedAttr(".nrope") ) return 0;
	if(!hasNamedData(".rope") ) return 0;
	if(!hasNamedAttr(".nind") ) return 0;
	if(!hasNamedData(".ind") ) return 0;
	if(!hasNamedAttr(".nleaf") ) return 0;
	if(!hasNamedData(".leaf") ) return 0;
	if(!hasNamedAttr(".nnode") ) return 0;
	if(!hasNamedData(".node") ) return 0;
	if(!hasNamedAttr(".bbx") ) return 0;
	return 1;
}

}