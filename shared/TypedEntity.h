/*
 *  TypedEntity.h
 *  
 *
 *  Created by jian zhang on 10/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class TypedEntity {
public:
    enum TypeEntries {
        TTriangleMesh = 0,
        TPatchMesh = 1,
        TUnknown = 2
    };
    
	TypedEntity();
	void setEntityType(TypeEntries val);
	unsigned entityType() const;
	
	bool isTriangleMesh() const;
	bool isPatchMesh() const;
private:	
	unsigned m_type;
};