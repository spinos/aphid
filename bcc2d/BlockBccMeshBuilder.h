/*
 *  BlockBccMeshBuilder.h
 *  bcc
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AOrientedBox.h>
class BlockBccMeshBuilder {
public:
	BlockBccMeshBuilder();
	virtual ~BlockBccMeshBuilder();
	
	void build(const AOrientedBox & ob, 
				int gx, int gy, int gz);
protected:

private:
};