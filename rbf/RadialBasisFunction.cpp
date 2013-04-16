/*
 *  RadialBasisFunction.cpp
 *  rbf
 *
 *  Created by jian zhang on 4/12/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include "RadialBasisFunction.h"

RadialBasisFunction::RadialBasisFunction() {m_tau = 1.f;}
RadialBasisFunction::~RadialBasisFunction() 
{
	delete[] m_xi;
	delete[] m_res;
	for(std::vector<float *>::iterator i = m_ci.begin(); i != m_ci.end(); ++i)
		delete[] *i;
	for(std::vector<float *>::iterator i = m_wi.begin(); i != m_wi.end(); ++i)
		delete[] *i;
}

void RadialBasisFunction::setTau(float tau)
{
	m_tau = tau;
}

float RadialBasisFunction::Phir(float r) const
{
	return exp(-r/m_tau);
}

void RadialBasisFunction::create(unsigned num)
{
	m_numNodes = num;
	m_xi = new Vector3F[num];
	m_res = new float[num];
	for(unsigned i = 0; i < num; i++) {
		m_ci.push_back(new float[num]);
		m_wi.push_back(new float[num]);
	}
	for(unsigned j = 0; j < num; j++) {
		for(unsigned i = 0; i < num; i++) {
			if(i == j) m_ci[j][i] = 1.f;
			else  m_ci[j][i] = 0.f;
		}
	}
}

void RadialBasisFunction::setXi(unsigned idx, Vector3F xi)
{
	m_xi[idx] = xi;
}

void RadialBasisFunction::computeWeights()
{
	Eigen::MatrixXd A(m_numNodes, m_numNodes);
	for(unsigned j=0; j < m_numNodes; j++) {
		for(unsigned i=0; i < m_numNodes; i++) {
			if(i == j)
				A(j, i) = 1.0;
			else
				A(j, i) = Phir((m_xi[i] - m_xi[j]).length());
		}
	}
	
	for(unsigned i=0; i < m_numNodes; i++) 
		weightI(i, A);
}

void RadialBasisFunction::solve(Vector3F x) const
{
	for(unsigned j = 0; j < m_numNodes; j++) {
		m_res[j] = 0.f;
		for(unsigned i=0; i < m_numNodes; i++) {
			m_res[j] += m_wi[j][i] * Phir((x - m_xi[i]).length());
		}
	}
	
	float sum = 0;
	for(unsigned j = 0; j < m_numNodes; j++) {
		if(m_res[j] > 0.0)
			sum += m_res[j];
		else
			m_res[j] = 0.0;
	}
	
	for(unsigned j = 0; j < m_numNodes; j++) {
		m_res[j] /= sum;
	}
}

void RadialBasisFunction::weightI(unsigned idx, Eigen::MatrixXd & A)
{
	Eigen::VectorXd b(m_numNodes);
	for(unsigned i=0; i < m_numNodes; i++) {
		b(i) = m_ci[idx][i];
	}
	
	Eigen::VectorXd x = A.lu().solve(b);
	
	for(unsigned i=0; i < m_numNodes; i++) {
		m_wi[idx][i] = x(i);
	}
}

unsigned RadialBasisFunction::getNumNodes() const
{
	return m_numNodes;
}
	
Vector3F RadialBasisFunction::getXi(unsigned idx) const
{
	return m_xi[idx];
}

float RadialBasisFunction::getResult(unsigned idx) const
{
	return m_res[idx];
}
