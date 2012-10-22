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
	TypedEntity();
	void setMeshType();
	bool isMesh() const;
private:	
	unsigned m_type;
};