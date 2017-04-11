/*
 *  PlantPiece.h
 *  garden
 *
 *  holds geometry, tm for branching
 *
 *  Created by jian zhang on 4/15/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_PLANT_PIECE_H
#define GAR_PLANT_PIECE_H

#include <math/Matrix44F.h>
#include <vector>

namespace aphid {
class ATriangleMesh;
}

class PlantPiece {
/// local space parent
	aphid::Matrix44F m_tm;
	
typedef std::vector<PlantPiece * > ChildListTyp;
	ChildListTyp m_childPieces;
	PlantPiece * m_parentPiece;
	aphid::ATriangleMesh * m_geom;
	
public:
	PlantPiece(PlantPiece * parent = NULL);
	virtual ~PlantPiece();
	
	void addBranch(PlantPiece * c);
	
	void setTransformMatrix(const aphid::Matrix44F &tm);
	const aphid::Matrix44F & transformMatrix() const;
	
	int numBranches() const;
	const PlantPiece * branch(const int & i) const;
	
	void setGeometry(aphid::ATriangleMesh * geom);
	const aphid::ATriangleMesh * geometry() const;
	
protected:

private:
};

#endif