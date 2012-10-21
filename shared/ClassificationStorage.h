/*
 *  ClassificationStorage.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class ClassificationStorage {
public:
	ClassificationStorage();
	~ClassificationStorage();
	
	void clear();
	
	void setPrimitiveCount(unsigned size);
	void set(unsigned index, int value);
	int get(unsigned index) const;
	unsigned size() const;
private:
	char *m_buffer;
	unsigned m_bufferSize;
};