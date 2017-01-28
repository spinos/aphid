/*
 *  DrawTetrahedron.h
 *  
 *
 *  Created by jian zhang on 2/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

namespace aphid {

namespace cvx {

class Tetrahedron;

}

class DrawTetrahedron {

public:

	DrawTetrahedron();
	virtual ~DrawTetrahedron();
	
protected:
	void drawAWireTetrahedron(const cvx::Tetrahedron & tet) const;
    void drawASolidTetrahedron(const cvx::Tetrahedron & tet) const;
    void drawAShrinkSolidTetrahedron(const cvx::Tetrahedron & tet,
                            const float & shrink) const;
    
private:
	
};

}