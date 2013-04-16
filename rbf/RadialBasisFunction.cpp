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
	delete[] m_ci;
	delete[] m_wi;
}

void RadialBasisFunction::setTau(float tau)
{
	m_tau = tau;
}

float RadialBasisFunction::Phir(float r) const
{
	return exp(-(r*r)/(m_tau * m_tau));
}

void RadialBasisFunction::create(unsigned num)
{
	m_numNodes = num;
	m_xi = new Vector3F[num];
	m_ci = new float[num];
	m_wi = new float[num];
}

void RadialBasisFunction::setXi(unsigned idx, Vector3F xi)
{
	m_xi[idx] = xi;
}

void RadialBasisFunction::setCi(unsigned idx, float ci)
{
	m_ci[idx] = ci;
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
	
	Eigen::VectorXd b(m_numNodes);
	for(unsigned i=0; i < m_numNodes; i++) {
		b(i) = m_ci[i];
	}
	
	Eigen::VectorXd x = A.lu().solve(b);
	
	std::cout<<"w "<<x;
	for(unsigned i=0; i < m_numNodes; i++) {
		m_wi[i] = x(i);
	}
}

float RadialBasisFunction::solve(Vector3F x) const
{
	float res = 0.f;
	for(unsigned i=0; i < m_numNodes; i++) {
		res += m_wi[i] * Phir((x - m_xi[i]).length());
	}
	return res;
}

