/*
 *  RedBlueRefine.h
 *  
 *
 *  Created by jian zhang on 7/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "triangulation.h"
#include "tetrahedron_graph.h"

namespace ttg {

/* c-------d
 *  \    / |
 *   \  /  |
 *    a----b
 *  4 vertices in a b c d order
 * 
 * c-------d
 *    a----b
 *  2 red edge
 *
 * c      d
 *  \    /
 *   \  / 
 *    a  
 * c     d
 *  \   /
 *   \ /
 *    b
 *  4 blue edge
 * 
 *  +- ++ -- +0 -0 00
 *  split edges has distance, sign change, zero end does not split
 * 
 *  1 red (+ - 0 0) or 1 blue (- 0 0 +) split into 2 tetra
 *  1 red and 1 blue (+ - 0 +) split into 3 tetra
 *  2 blue one side (- 0 + +) split into 4 tetra
 *  2 blue both side (- - 0 +) split into 4 tetra
 *  1 red 2 blue down (- + - -) up (- - - +) split into 6 tetra
 *  cannot have 2 red split
 *  cannot have 3 blue split
 *  cannot have 1 red 4 blue
 *  2 red 2 blue (+ - + -) split into 8 tetra
 *  4 blue (- - + +) split into 8 tetra
 *  auxoliary 2 red 4 blue
 *  add 6 vertice at max
 */

class RedBlueRefine {

	ITetrahedron m_tet[8];
	int m_a, m_b, m_c, m_d;
/// ind to split v, 
/// -1 : no split
///  0 : wait for input
/// >0 : valid
	int m_red[2];
	int m_blue[4];
	int m_N;
	
	enum SplitOption {
		SpNone = 0,
		SpOneRed = 1,
		SpOneRedOneBlue = 2,
		SpOneRedDownTwoBlue = 3,
		SpOneRedUpTwoBlue = 4,
		SpTwoRedTwoBlue = 5,
		SpOneBlue = 6,
		SpTwoBlueUp = 7,
		SpTwoBlueDown = 8,
		SpFourBlue = 9
	};
	
	SplitOption m_opt;
	
public:
	RedBlueRefine();
	
/// ind to vertices
	void set(int a, int b, int c, int d);
/// distance of vertices, determine six edge splits
	void evaluateDistance(float a, float b, float c, float d);
	const int & numTetra() const;
	const ITetrahedron * tetra(int i) const;
	
	void splitRedEdge(int i, int v);
	void splitBlueEdge(int i, int v);
	bool needSplitRedEdge(int i);
	bool needSplitBlueEdge(int i);
	
	aphid::Vector3F splitPos(float a, float b,
						const aphid::Vector3F & pa,
						const aphid::Vector3F & pb) const;
	
	void refine();
	void verbose() const;
	bool hasOption() const;
	
private:
/// sign changes
	bool splitCondition(const float & a,
					const float & b) const;
/// split not on front but needed to prevent frustum
	void auxiliarySplit();
	void oneRedRefine();
	void oneBlueRefine();
	void oneRedOneBlueRefine();
	void oneRedUpTwoBlueRefine();
	void oneRedDownTwoBlueRefine();
	void twoRedTwoBlueRefine();
	void fourBlueRefine();
	void twoBlueUpRefine();
	void twoBlueDownRefine();
	void splitRed(int i, ITetrahedron & t0, ITetrahedron & t1,
								int v);
	void splitBlue(int i, ITetrahedron & t0, ITetrahedron & t1,
								int v);
};

}
