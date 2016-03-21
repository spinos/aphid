/*
 *  QuadraticErrorFunction.h
 *  
 *
 *  Created by jian zhang on 3/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  E[x] = Sigma i (ni . (x - pi) )^2
 *  minimize Ax - b
 *  A is n x 3 matrix whose rows are ni
 *  b is n vector whose entries are ni . pi
 *  x is 3 vector 
 *  reference https://inst.eecs.berkeley.edu/~ee127a/book/login/def_pseudo_inv.html
 *  solve Ax = b by pseudo-inversion of A
 */
#include "linearMath.h"
#include <iostream>

namespace aphid {

template<typename T, int Dim>
class QuadraticErrorFunction {

/// A is full column rank, n <= m
	lfr::DenseMatrix<T> m_A;
	lfr::DenseVector<T> m_b;
	lfr::SvdSolver<T> m_svd;
	lfr::DenseVector<T> m_x;
/// left inverse of A
	lfr::DenseMatrix<T> m_Astar;
	
public:
	QuadraticErrorFunction();
	virtual ~QuadraticErrorFunction();
	
	void create(int nrows);
	
	void copyARow(int irow, const T * src);
	void copyB(int i, const T * src);
	bool compute();
	
	const lfr::DenseVector<T> & x() const;
	
private:

};

template<typename T, int Dim>
QuadraticErrorFunction<T, Dim>::QuadraticErrorFunction()
{}

template<typename T, int Dim>
QuadraticErrorFunction<T, Dim>::~QuadraticErrorFunction()
{}

template<typename T, int Dim>
void QuadraticErrorFunction<T, Dim>::create(int nrows)
{
	m_A.resize(nrows, Dim);
	m_Astar.resize(Dim, nrows);
	m_b.resize(nrows);
	m_x.resize(Dim);
}

template<typename T, int Dim>
void QuadraticErrorFunction<T, Dim>::copyARow(int irow, const T * src)
{
	int i = 0;
	for(;i<Dim;++i) {
		m_A.column(i)[irow] = src[i];
	}
}
	
template<typename T, int Dim>
void QuadraticErrorFunction<T, Dim>::copyB(int i, const T * src)
{
	m_b.v()[i] = *src;
}
	
template<typename T, int Dim>
bool QuadraticErrorFunction<T, Dim>::compute()
{
	m_svd.compute(m_A);
	
/// last singular value
	if(m_svd.S().v()[Dim-1] < 1e-3) {
		std::cout<<"\n s"<<m_svd.S();
		std::cout<<"\n A is near singular, no solution";
		return false;
	}
	
	std::cout<<"\n s"<<m_svd.S();
	std::cout<<"\n u"<<m_svd.U();
	std::cout<<"\n v"<<m_svd.Vt();
	
	lfr::DenseMatrix<T> S1(m_svd.U().numRows(), Dim);
	S1.setZero();
	for(int i=0;i<Dim;++i)
		S1.column(i)[i] = (T)1.0 / (m_svd.S()[i] + 1e-15);
		
	std::cout<<"\n S^-1 "<< S1;
		
	lfr::DenseMatrix<T> VS1(Dim, Dim);
	m_svd.Vt().transMult(VS1, S1);
	
	std::cout<<"\n V^T S^-1 "<< VS1;

	VS1.multTrans(m_Astar, m_svd.U() );
	std::cout<<"\n A* "<< m_Astar;
	
	m_Astar.mult(m_x, m_b);
	
	return true;
}

template<typename T, int Dim>
const lfr::DenseVector<T> & QuadraticErrorFunction<T, Dim>::x() const
{ return m_x; }

}