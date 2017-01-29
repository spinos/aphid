/*
 *  RedBlueRefine.h
 *  
 *
 *  Created by jian zhang on 7/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_TTG_RED_BLUE_REFINE_H
#define APH_TTG_RED_BLUE_REFINE_H

#include "tetrahedron_graph.h"

namespace aphid {

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
 *  estimate normal by six edge value changes
 *  look for triangles with three front vertices
 *  have 2 front triangles at max
 */

class RedBlueRefine {

	Vector3F m_p[10];
	Vector3F m_normal;
	float m_fa, m_fb, m_fc, m_fd;
	float m_normalLen;
	ITetrahedron m_tet[8];
	IFace m_frontTri[2];
	int m_a, m_b, m_c, m_d;
	int m_N, m_numTri;
/// ind to split v, 
/// -1 : no split
///  0 : wait for input
/// >0 : valid
	int m_red[2];
	int m_blue[4];
	
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
	void estimateNormal(const Vector3F & a,
							const Vector3F & b,
							const Vector3F & c,
							const Vector3F & d);
	const int & numTetra() const;
	const ITetrahedron * tetra(int i) const;
	
	void splitRedEdge(int i, int v, const Vector3F & p);
	void splitBlueEdge(int i, int v, const Vector3F & p);
	bool needSplitRedEdge(int i);
	bool needSplitBlueEdge(int i);
	
	Vector3F splitPos(float a, float b,
						const Vector3F & pa,
						const Vector3F & pb) const;
	
	void refine();
	void verbose() const;
	bool hasOption() const;
	bool checkTetraVolume() const;
	bool hasNormal() const;
	const Vector3F & normal() const;
	const int & numFrontTriangles() const;
	const IFace * frontTriangle(const int & i) const;
	
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
	int pInd(int i) const;
///  1: a to b
///  0: 0
/// -1: b to a
	float edgeDir(const float & a, const float & b) const;
	void findOneTriangle();
	void buildTriangle(IFace & t,
						const int & va, const int & vb, const int & vc);
/// 0,1 0,2 3,2 3,1 
	void findTwoBlue(int & b1, int & b2) const;
	
};

}

}
#endif
