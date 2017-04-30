/*
 *  CalamusSkin.h
 *  mallard
 *
 *  Created by jian zhang on 12/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <CollisionRegion.h>
#include <boost/scoped_array.hpp>
class MlCalamus;
class MlCalamusArray;
class MlCluster;
class BaseSphere;
class CalamusSkin : public CollisionRegion {
public:
	struct FloodTable {
		FloodTable() {}
		FloodTable(unsigned i) {
			reset(i);
		}
		
		void reset(unsigned i) {
			faceIdx = i;
			dartBegin = dartEnd = 0;
		}
		
		unsigned faceIdx, dartBegin, dartEnd;
	};
	
	CalamusSkin();
	virtual ~CalamusSkin();
	
	void cleanup();
	
	void setBodyMesh(AccPatchMesh * mesh);
	void getPointOnBody(MlCalamus * c, Vector3F &p) const;
	void getNormalOnBody(MlCalamus * c, Vector3F &p) const;
	
	void tangentSpace(MlCalamus * c, Matrix33F & frm) const;
	void rotationFrame(MlCalamus * c, const Matrix33F & tang, Matrix33F & frm) const;
	void calamusSpace(MlCalamus * c, Matrix33F & frm) const;
	MlCalamusArray * getCalamusArray() const;
	MlCalamus * getCalamus(unsigned idx) const;
	
	void clearFeather();
	void setNumFeathers(unsigned num);
	unsigned numFeathers() const;
	void addFeather(MlCalamus & ori);
	void zeroFeather();
	void reduceFeather(unsigned num);
	
	void touchBy(MlCalamus * c, const Vector3F & pos, const Matrix33F & frm);
	
	void createFaceCluster();
	void computeFaceClustering();
	void computeClusterSamples();
	char useClusterSamples(unsigned faceIdx, unsigned perFaceIdx, MlCalamus * c, unsigned ci);
	void getClustering(unsigned idx, std::vector<Vector3F> & dst);
	unsigned clusterK(unsigned faceIdx) const;
	
	void createFaceCalamusIndirection();
	void resetFaceCalamusIndirection();
	void computeFaceCalamusIndirection();
	void faceCalamusBeginEnd(unsigned faceIdx, unsigned & begin, unsigned & end) const;
protected:
	bool isPointTooCloseToExisting(const Vector3F & pos, float minDistance);

private:
	MlCalamusArray * m_calamus;
	FloodTable * m_faceCalamusTable;
	boost::scoped_array<MlCluster> m_perFaceCluster;
	unsigned m_numFeather;
};