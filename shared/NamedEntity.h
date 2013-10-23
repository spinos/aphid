/*
 *  NamedEntity.h
 *  eulerRot
 *
 *  Created by jian zhang on 10/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <string>

class NamedEntity {
public:
	NamedEntity();
	virtual ~NamedEntity();
	
	void setName(const std::string & name);
	void setName(const std::string & name, int i);
	std::string name() const;
	std::string NamedEntity::particalName() const;
	
	void setIndex(unsigned idx);
	unsigned index() const;
private:
	std::string m_name;
	unsigned m_index;
};