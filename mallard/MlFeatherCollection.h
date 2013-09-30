/*
 *  MlFeatherCollection.h
 *  mallard
 *
 *  Created by jian zhang on 10/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <map>
class MlFeather;

class MlFeatherCollection {
public:
	MlFeatherCollection();
	virtual ~MlFeatherCollection();
	
	void clearFeatherExamples();
	
	MlFeather * addFeatherExample();
	unsigned numFeatherExamples() const;
	void selectFeatherExample(unsigned x);
	
	unsigned selectedFeatherExampleId() const;
	MlFeather * selectedFeatherExample();
	MlFeather * featherExample(unsigned idx);
	
	void initializeFeatherExample();
	
private:
	std::map<unsigned, MlFeather *> m_feathers;
	unsigned m_selectedFeatherId;
};

