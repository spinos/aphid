/*
 *  kmean.h
 *  
 *	cluster N D-dimensional points into K groups
 *
 *  Created by jian zhang on 12/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MATH_K_MEAN_H
#define APH_MATH_K_MEAN_H

#include <math/linearMath.h>
#include <boost/scoped_array.hpp>

namespace aphid {

template<typename T>
class KMeansClustering2 {

/// size D-by-K, stored columnwise
	DenseMatrix<T> m_centroids;
	DenseMatrix<T> m_groupMean;
/// ind to group per point, size N
	boost::scoped_array<int> m_groupInd;
/// count points in group, size K
	boost::scoped_array<int> m_groupCount;
	int m_K, m_N, m_D;
	T m_separateDistance;
	
public:
	KMeansClustering2();
	virtual ~KMeansClustering2();
	
	void setKND(const int & k,
				const int & n,
				const int & d);
	
/// points stored rowwise, size N-by-D
	bool compute(const DenseMatrix<T> & points);
	
	const DenseMatrix<T> & groupCentroids() const;
/// n indices
	const int * groupIndices() const;
	const int & K() const;
	void getGroupCentroid(DenseVector<T> & d, 
							const int & i) const;
							
	void setSeparateDistance(const T & x);
	
protected:
	void setGroupCentroid(const DenseVector<T> & d, 
							const int & i);
	
private:
/// get i-th row point
	void getXi(DenseVector<T> & dst, 
				const DenseMatrix<T> & points,
				const int & idx) const;
	bool assignPointsToGroup(const DenseMatrix<T> & points);
	int closestGroupTo(const DenseVector<T> & apoint) const;
	T moveCentroid();
	bool farEnoughToPreviousCentroids(const DenseVector<T> & pnt,
							const int & lastI) const;
	bool assignToOneGroup(const DenseMatrix<T> & points);
	bool assignToEachGroup(const DenseMatrix<T> & points);
								
};

template<typename T>
KMeansClustering2<T>::KMeansClustering2()
{ m_separateDistance = 0.4; }

template<typename T>
KMeansClustering2<T>::~KMeansClustering2()
{}

template<typename T>
void KMeansClustering2<T>::setSeparateDistance(const T & x)
{ m_separateDistance = x; }

template<typename T>
void KMeansClustering2<T>::setKND(const int & k,
				const int & n,
				const int & d)
{
	m_N = n;
	m_groupInd.reset(new int[m_N]);
	
	m_K = k;
	m_groupCount.reset(new int[m_K]);
	
	m_D = d;
	m_centroids.resize(m_D, m_K);
	m_groupMean.resize(m_D, m_K);
}

template<typename T>
void KMeansClustering2<T>::getXi(DenseVector<T> & dst, 
				const DenseMatrix<T> & points,
				const int & idx) const
{
	points.extractRowData(dst.v(), idx);
}

template<typename T>
bool KMeansClustering2<T>::farEnoughToPreviousCentroids(const DenseVector<T> & pnt,
							const int & lastI) const
{
	T minD = 1e20;
	for(int i=0;i <= lastI;++i) {
		DenseVector<T> gcen(m_D);
		gcen.copyData(m_centroids.column(i) );
		
		T diff = (gcen - pnt).norm();
		if(minD > diff) {
			minD = diff;
		}
	}
	return minD > m_separateDistance;
}

template<typename T>
bool KMeansClustering2<T>::assignToOneGroup(const DenseMatrix<T> & points)
{
	for(int i=0;i<m_N;++i) {
		m_groupInd[i] = 0;
	}
	m_groupCount[0] = m_N;
/// first data as only group centroid
	DenseVector<T> apnt(m_D);
	getXi(apnt, points, 0);
	m_centroids.copyColumn(0, apnt.c_v() );
	return true;
}
	
template<typename T>
bool KMeansClustering2<T>::assignToEachGroup(const DenseMatrix<T> & points)
{
	DenseVector<T> apnt(m_D);
	for(int i=0;i<m_N;++i) {
		m_groupInd[i] = i;
		getXi(apnt, points, i);
		m_centroids.copyColumn(i, apnt.c_v() );
	}
	for(int i=0;i<m_K;++i) {
		m_groupCount[i] = 1;
	}
	return true;
}

template<typename T>
bool KMeansClustering2<T>::compute(const DenseMatrix<T> & points)
{
	if(m_K < 2) {
		return assignToOneGroup(points);
	} else if (m_K >= m_N) {
		return assignToEachGroup(points);
	}
	
	DenseVector<T> apnt(m_D);
	
/// initial guess
/// go through point select one with large difference 
/// to previously selected ones

	getXi(apnt, points, 0);
	m_centroids.copyColumn(0, apnt.c_v() );
	int nsel = 1;
	
	for(int i=1;i<m_N;++i) {
		getXi(apnt, points, i);
		if(farEnoughToPreviousCentroids(apnt, i) ) {
			m_centroids.copyColumn(nsel, apnt.c_v() );
			//std::cout<<"\n select point "<<i<<" as centroid "<<nsel;
			nsel++;
			if(nsel==m_K) {
				break;
			}
		}
	}
	
	if(nsel < m_K) {
		std::cout<<"\n kmean cannot find enough deviation to have "<<m_K<<" groups "
				<<"\n K = "<<nsel;
		m_K = nsel;
	}
	
/// all to group 0
	for(int i=0;i<m_N;++i) {
		m_groupInd[i] = 0;
	}
	
	int i=0;
	for(;i<29;++i) {
		bool changed = assignPointsToGroup(points);
		moveCentroid();
		if(!changed) {
			break;
		}
		
	}
	
	//std::cout<<"\n kmean finish after "<<i<<" updates ";
	//std::cout.flush();
	return true;
	
}

template<typename T>
int KMeansClustering2<T>::closestGroupTo(const DenseVector<T> & apoint) const
{
	
	T maxDist = 1e20;
	int ind = 0;
	for(int i=0;i<m_K;++i) {
		const DenseVector<T> gcen(m_centroids.column(i), m_D);
		T dist = gcen.distanceTo(apoint);
		if(maxDist > dist) {
			maxDist = dist;
			ind = i;
		}
	}
	return ind;
}

template<typename T>
bool KMeansClustering2<T>::assignPointsToGroup(const DenseMatrix<T> & points)
{
	m_groupMean.setZero();
	
	for(int i=0;i<m_K;++i) {
		m_groupCount[i] = 0;
	}
	
	DenseVector<T> apnt(m_D);
	bool changed = false;
	for(int i=0;i<m_N;++i) {
		getXi(apnt, points, i);
		
		const int closestG = closestGroupTo(apnt);
		
		DenseVector<T> gmean(m_groupMean.column(closestG), m_D);
		gmean.add(apnt);
		
		m_groupCount[closestG]++;
		
		if(m_groupInd[i] != closestG) {
			m_groupInd[i] = closestG;
			changed = true;
		}
	}
	return changed;
}

template<typename T>
T KMeansClustering2<T>::moveCentroid()
{
	T sumMoved = 0;
	
	DenseVector<T> vmean(m_D);
		
	for(int i=0;i<m_K;++i) {
		
		vmean.copyData(m_groupMean.column(i) );
		vmean.scale((T)1.0 / (T)m_groupCount[i]);
		
		DenseVector<T> gcen(m_centroids.column(i), m_D);
		
		DenseVector<T> diff = gcen - vmean;
		
		sumMoved += diff.norm();
		
		setGroupCentroid(vmean, i);
		
	}
	
	return (sumMoved/ (T)m_K);
}

template<typename T>
const DenseMatrix<T> & KMeansClustering2<T>::groupCentroids() const
{ return m_centroids; }

template<typename T>
const int * KMeansClustering2<T>::groupIndices() const
{ return &m_groupInd[0]; }

template<typename T>
const int & KMeansClustering2<T>::K() const
{ return m_K; }

template<typename T>
void KMeansClustering2<T>::getGroupCentroid(DenseVector<T> & d, 
							const int & i) const
{
	d.resize(m_D);
	d.copyData(m_centroids.column(i) );
}

template<typename T>
void KMeansClustering2<T>::setGroupCentroid(const DenseVector<T> & d, 
							const int & i)
{
	m_centroids.copyColumn(i, d.c_v() );
}

}

#endif
