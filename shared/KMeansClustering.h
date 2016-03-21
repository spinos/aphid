/*
 *  KMeansClustering.h
 *  aphid
 *
 *  Created by jian zhang on 12/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Vector3F.h>
#include <VectorArray.h>

namespace aphid {

namespace kmean {

template<typename T>
class Cluster {

	sdb::VectorArray<T> m_object;
	sdb::VectorArray<int> m_membership;
	sdb::VectorArray<T> m_group;
	sdb::VectorArray<T> m_cur;
	sdb::VectorArray<int> m_count;
	
	int m_n, m_k;
	bool m_toNormalize;
	
public:
	Cluster();
	virtual ~Cluster();
	
	void setN(int n);
	void setK(int k);
	void setToNormalize(bool x);
	
	sdb::VectorArray<T> & object();
	
	void compute();
	void printResult() const;
	
private:
	float update();
	void validateResult();
	
};

template<typename T>
Cluster<T>::Cluster() : m_toNormalize(false)
{}

template<typename T>
Cluster<T>::~Cluster()
{}

template<typename T>
void Cluster<T>::setN(int n)
{
	m_object.clear();
	m_membership.clear();
	
	int i=0;
	for(;i<n;++i) {
		m_object.insert();
		m_membership.insert(-1);
	}
	
	m_n = n;
}

template<typename T>
void Cluster<T>::setK(int k)
{
	m_group.clear();
	int i=0;
	for(;i<k;++i) {
		m_group.insert();
		m_cur.insert();
		m_count.insert();
	}
	
	m_k = k;
}

template<typename T>
void Cluster<T>::setToNormalize(bool x)
{ m_toNormalize = x; }
	
template<typename T>
sdb::VectorArray<T> & Cluster<T>::object()
{ return m_object; }

template<typename T>
void Cluster<T>::compute()
{
/// randomly pick k objects as initial group
/// in case first k objects are closely located
	int i=0;
	for(;i<m_k;++i) {
		*m_group[i] = *m_object[rand() % m_n];
		// std::cout<<"\n group["<<i<<"] "<<*m_group[i];
	}
	
/// update until no more changes
	float delta = 1.0;
	while(delta > 1e-3f) {
		delta = update();
	}
	
	validateResult();
}
	
template<typename T>
float Cluster<T>::update()
{
	int i, j;
/// clear groups to update
	for(i=0;i<m_k;++i) {
		m_cur[i]->setZero();
		*m_count[i] = 0;
	}
	
	float delta = 0.f;
	float dist, minDist;
	int mi;
	for(i=0;i<m_n;++i) {
		minDist = 1e27f;
/// find the nearest cluster
		for(j=0;j<m_k;++j) {
			dist = m_object[i]->distanceTo(*m_group[j]);
			if(minDist > dist) {
				minDist = dist;
				mi = j;
			}
		}
/// membership changed		
		if(*m_membership[i] != mi) {
			delta += 1.f;
			*m_membership[i] = mi;
		}

		// std::cout<<"\n add "<<i<<" to group "<<mi;
/// assign to group 		
		*m_cur[mi] += *m_object[i];
		(*m_count[mi])++;
		
	}
	
	for(i=0;i<m_k;++i) {
		if(*m_count[i] > 1) 
			*m_cur[i] /= (float)*m_count[i];
			
		if(*m_count[i] > 0 && m_toNormalize)
			m_cur[i]->normalize();
		
		*m_group[i] = *m_cur[i];
		
		//std::cout<<"\n group["<<i<<"] "<<*m_group[i]
		//	<<" count "<<*m_count[i];
	}
	
	// std::cout<<"\n membership changed "<<delta;
	return delta / (float)m_n;
}

template<typename T>
void Cluster<T>::printResult() const
{
	for(int i=0;i<m_k;++i) {
		std::cout<<"\n group["<<i<<"] "<<*m_group[i]
			<<" n members "<<*m_count[i];
	}
}

template<typename T>
void Cluster<T>::validateResult()
{
/// eliminate empty group
	for(int i=0;i<m_k;++i) {
		if(*m_count[i] < 1) {
			*m_group[i] = *m_group[m_k-1];
			*m_count[i] = *m_count[m_k-1];
			m_k--;
		}
	}
}

}

class KMeansClustering {
public:
	KMeansClustering();
	virtual ~KMeansClustering();
	
	void preAssign();
	void assignToGroup(unsigned idx, const Vector3F & pos);
	float moveCentroids();
	void resetGroup();
	Vector3F groupCenter(unsigned idx) const;
	unsigned K() const;
	unsigned N() const;
	char isValid() const;
	unsigned group(unsigned idx) const;
	unsigned countPerGroup(unsigned idx) const;
	const Vector3F centroid(unsigned igroup) const;
protected:
    virtual void setK(const unsigned & k);
	virtual void setN(unsigned n);
	virtual void initialGuess(const Vector3F * pos);
	void setCentroid(unsigned idx, const Vector3F & pos);
	void setValid(char val);
	
private:
	Vector3F * m_centroid;
	Vector3F * m_sum;
	unsigned * m_group;
	unsigned * m_countPerGroup;
	unsigned m_k, m_n;
	char m_valid;
};

}