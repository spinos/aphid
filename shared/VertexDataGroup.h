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
	
	void createEntry(const std::string & name, unsigned num, short bpe);
	char hasEntry(const std::string & name) const;
	char * entry(const std::string & name);
	char * entry(const unsigned & idx, std::string & name, short & bpe);
	unsigned fpe(const std::string & name);
	
	unsigned numEntries() const;
private:
	struct NameAndType {
		std::string _name;
		short _type;
	};
	
	int entryIdx(const std::string & name) const;
	std::vector<NameAndType> m_names;
	std::vector<char *> m_entries;
};