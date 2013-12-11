/*
 *  KMeansClustering.cpp
 *  aphid
 *
 *  Created by jian zhang on 12/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "KMeansClustering.h"

KMeansClustering::KMeansClustering() : m_centroid(0), m_group(0), m_sum(0), m_countPerGroup(0), m_valid(0) {}

KMeansClustering::~KMeansClustering() 
{
	if(m_centroid) delete[] m_centroid;
	if(m_sum) delete[] m_sum;
	if(m_group) delete[] m_group;
	if(m_countPerGroup) delete[] m_countPerGroup;
}

void KMeansClustering::setK(unsigned k)
{
	m_k = k;
	if(m_centroid) delete[] m_centroid;
	m_centroid = new Vector3F[k];
	if(m_sum) delete[] m_sum;
	m_sum = new Vector3F[k];
	if(m_countPerGroup) delete[] m_countPerGroup;
	m_countPerGroup = new unsigned[k];
}

void KMeansClustering::setN(unsigned n)
{
	m_n = n;
	if(m_group) delete[] m_group;
	m_group = new unsigned[n];
}

void KMeansClustering::setInitialGuess(unsigned idx, const Vector3F & pos)
{
	m_centroid[idx] = pos;
}

void KMeansClustering::preAssign()
{
	for(unsigned i = 0; i < m_k; i++) m_sum[i].setZero();
}

void KMeansClustering::assignToGroup(unsigned idx, const Vector3F & pos)
{
	unsigned g = 0;
	float minD = 10e8;
	float d;
	for(unsigned i = 0; i < m_k; i++) {
		d = Vector3F(pos, m_centroid[i]).length();
		if(d < minD) {
			g = i;
			minD = d;
		}
	}
	m_group[idx] = g;
	m_sum[g] += pos;
}

float KMeansClustering::moveCentroids()
{
	float delta = 0.f;
	for(unsigned i = 0; i < m_k; i++) m_countPerGroup[i] = 0;
	for(unsigned i = 0; i < m_n; i++) {
		m_countPerGroup[m_group[i]] += 1;
	}
	for(unsigned i = 0; i < m_k; i++) {
		m_sum[i] *= 1.f/(float)m_countPerGroup[i];
		delta += Vector3F(m_centroid[i], m_sum[i]).length();
		m_centroid[i] = m_sum[i];
	}
	return delta;
}

void KMeansClustering::resetGroup()
{
	for(unsigned i = 0; i < m_n; i++) m_group[i] = i;
}

Vector3F KMeansClustering::groupCenter(unsigned idx) const
{
	return m_centroid[m_group[idx]];
}

unsigned KMeansClustering::group(unsigned idx) const
{
	return m_group[idx];
}

unsigned KMeansClustering::countPerGroup(unsigned idx) const
{
	return m_countPerGroup[idx];
}

unsigned KMeansClustering::K() const
{
	return m_k;
}

char KMeansClustering::isValid() const
{
	return m_valid;
}

void KMeansClustering::setValid(char val)
{
	m_valid = val;
}
