#ifndef APH_H_GROUP_H
#define APH_H_GROUP_H

/*
 *  HGroup.h
 *  helloHdf
 *
 *  Created by jian zhang on 6/12/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "HObject.h"

namespace aphid {

class HGroup : public HObject {
public:
	HGroup(const std::string & path);
	virtual ~HGroup() {}
	
    bool isOpened() const;
	virtual char create();
	virtual char open();
	virtual void close();
	virtual int objectType() const;
};

}
#endif        //  #ifndef HGROUP_H
