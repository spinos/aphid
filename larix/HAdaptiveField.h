/*
 *  HAdaptiveField.h
 *  aphid
 *
 *  Created by jian zhang on 8/5/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HField.h"

class AdaptiveField;

class HAdaptiveField : public HField {
public:
	HAdaptiveField(const std::string & path);
	virtual ~HAdaptiveField();
	
	virtual char verifyType();
	virtual char save(AdaptiveField * fld);
	virtual char load(AdaptiveField * fld);
protected:

private:

};