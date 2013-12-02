/*
 *  VertexDataGroup.h
 *  aphid
 *
 *  Created by jian zhang on 12/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>

class VertexDataGroup {
public:
	VertexDataGroup();
	virtual ~VertexDataGroup();
	
	void createEntry(const std::string & name, unsigned num, unsigned fpe);
	char hasEntry(const std::string & name) const;
	float * entry(const std::string & name);
	float * entry(const unsigned & idx, std::string & name, unsigned & fpe);
	unsigned fpe(const std::string & name);
	
	unsigned numEntries() const;
private:
	struct NameAndType {
		std::string _name;
		unsigned _type;
	};
	
	int entryIdx(const std::string & name) const;
	std::vector<NameAndType> m_names;
	std::vector<float *> m_entries;
};