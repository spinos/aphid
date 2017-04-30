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
class MSelectionList;

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
	void connectInstanceGroup(MStringArray & instanceNames,
					const MObject& node,
					const int & iExample,
					const int & iL2Example);
	
private:
	int countInstanceTo(const MObject& node);
	int countInstanceToShrub(aphid::ProxyViz * viz,
					const MObject& node);
	int listInstanceTo(MStringArray & instanceNames,
					const MObject& node);
	int listInstanceToShrub(MStringArray & instanceNames,
					const MObject& node);
	void connectInstanceTo(MStringArray & instanceNames,
					MSelectionList & sels, 
					const MObject& node);
	void connectInstanceToShrub(MStringArray & instanceNames,
					MSelectionList & sels, 
					const MObject& node, 
					const int & iExample);
	
};
#endif