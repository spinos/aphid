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
class CollisionRegion;

class MlFeatherCollection {
public:
	MlFeatherCollection();
	virtual ~MlFeatherCollection();
	
	void clearFeatherExamples();
	
	MlFeather * addFeatherExample();
	MlFeather * addFeatherExampleId(unsigned idx);
	bool selectFeatherExample(unsigned x);
	
	unsigned selectedFeatherExampleId() const;
	bool removeSelectedFeatherExample();
	MlFeather * selectedFeatherExample();
	MlFeather * featherExample(unsigned idx);
	
	void initializeFeatherExample();
	
	MlFeather* firstFeatherExample();
	MlFeather* nextFeatherExample();
	bool hasFeatherExample();
	
	void setCollision(CollisionRegion * skin);
private:
	unsigned numFeatherExamples() const;
	bool featherIdExists(unsigned idx) const;
	unsigned usableId() const;
private:
	std::map<unsigned, MlFeather *> m_feathers;
	std::map<unsigned, MlFeather *>::iterator m_featherIt;
	unsigned m_selectedFeatherId;
};

