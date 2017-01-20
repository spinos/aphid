/*
 *  ReplacerWorks.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _REPLACER_WORKS_H
#define _REPLACER_WORKS_H

namespace aphid {
class ProxyViz;
}

class MObject;
class MStringArray;

class ReplacerWorks {

public:
	ReplacerWorks();
	virtual~ReplacerWorks();
	
protected:
	int listInstanceGroup(MStringArray & instanceNames,
					const MObject& node,
					const int & iExample);
	int countInstanceGroup(aphid::ProxyViz * viz,
					const MObject& node,
					const int & iExample);
private:
	int countInstanceTo(const MObject& node);
	int countInstanceToShrub(aphid::ProxyViz * viz,
					const MObject& node);
	int listInstanceTo(MStringArray & instanceNames,
					const MObject& node);
	int listInstanceToShrub(MStringArray & instanceNames,
					const MObject& node);
	
};
#endif