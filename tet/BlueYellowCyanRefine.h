#ifndef TTG_BLUE_YELLOW_CYAN_REFINE_H
#define TTG_BLUE_YELLOW_CYAN_REFINE_H

/*
 *  BlueYellowCyanRefine.h
 *  
 *
 *  Created by jian zhang on 7/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include "triangulation.h"
#include "tetrahedron_graph.h"

namespace ttg {

/* blue----cyan
 *  \    / |
 *   \  /  |
 *    red--yellow
 *
 * first tetra in red yellow blue cyan order
 * split blue yellow cyan edge once
 * into max four tetra
 *
 * split blue edge, added vertex as red of old tetra 
 * as blue of new tetra
 *
 * blue----cyan
 *  \    / |
 *   \  /  |
 *    + --yellow
 *
 * + ------cyan
 *  \    / |
 *   \  /  |
 *    red--yellow
 */

class BlueYellowCyanRefine {

	ITetrahedron m_tet[4];
	int m_N;
	
public:
	BlueYellowCyanRefine(int vred, int vyellow, int vblue, int vcyan);

	const int & numTetra() const;
	const ITetrahedron * tetra(int i) const;
	
	void splitYellow(int vyellow);
	void splitBlue(int vblue);
	void splitCyan(int vcyan);
	
private:
	ITetrahedron * lastTetra();
	
};

}
#endif        //  #ifndef TTG_BLUE_YELLOW_CYAN_REFINE_H
