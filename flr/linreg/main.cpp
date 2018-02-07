/*
 *  linear model estimation test
 */
 
#include <iostream>
#include <math/linearMath.h>
#include <math/miscfuncs.h>
#include <math/LinearRegression.h>
#include <QApplication>
#include <QDesktopWidget>
#include <QtCore>
#include "window.h"

using namespace aphid;

void testVVt()
{
	std::cout<<"\n test vvt";
	
	DenseVector<float> a(4);
	a[0] = 1.f;
	a[1] = 1.f;
	a[2] = 1.f;
	a[3] = 1.f;
	DenseVector<float> b(4);
	b[0] = 1.f;
	b[1] = 2.f;
	b[2] = 1.f;
	b[3] = 1.f;
	
	DenseMatrix<float> c(4,4);
	c.asVVt(a, b);
	std::cout<<"\n c "<<c;
}

/// Pixel History Linear Models for Real-Time Temporal Filter
/// f_i = B_i^T x_i
/// k is order of filter
/// d is number of coefficients
/// Y is k vector (y_1,...,y_k)^T k most recent signal
/// B is d+1 vector
/// x is d+1 vector (1,y_i) some function of Y?
/// P inverse covariance matrix d+1-by-d+1
void testEstimation()
{
	const int k = 4;
	const int d = 4;
	const int n = 16;
	
	DenseVector<float> Y(n);
	for(int i=0;i<n;++i) {
		Y[i] = .5f + .02f * i + RandomFn11() * .03f;
	}
	float yc = Y[0];

	std::cout<<"\n input signal Y"<<Y;
	
	DenseVector<float> beta;
	beta.create(d+1);
	beta.setZero();
	//beta[0] = yc;
	std::cout<<"\n b_0"<<beta;
	
	DenseVector<float> X;
	X.create(d+1);
	X.setZero();
	X[0] = 1.f;
	std::cout<<"\n X_0"<<X;
	
	DenseMatrix<float> P;
	P.create(d+1, d+1);
	P.setZero();
	P.addDiagonal(3000.f);
	std::cout<<"\n inverse covariance matrix P"<<P;
	
	DenseVector<float> xtp(d+1);
	DenseVector<float> Q(d+1);
	DenseMatrix<float> qxt(d+1,d+1);
	DenseMatrix<float> qxtp(d+1,d+1);
	
	const float lamda = .998f;
	
	for(int i=0;i<n;++i) {
		
		yc = 0.f;
		int begin = i - k + 1;
		if(begin < 0)
			begin = 0;
			
		std::cout<<"\n y["<<begin<<","<<i<<"]";
		int count = 0;
		for(int j=begin;j<=i;++j) {
			count++;
			yc += Y[j];
		}
		yc = yc / (float)count;
		std::cout<<"\n y_c "<<yc;	
		
/// x_i <- [1, z_i]
/// z_i is slope?
		for(int j=1;j<=count;++j)
			X[j] = Y[begin+j-1]; 
			
		std::cout<<"\n X_"<<i<<" "<<X;
		
		std::cout<<"\n B_"<<i<<" "<<beta;
		
		/// error_i <- y_i - f_i^hat		
		float err = Y[i] - beta.dot(X);
		std::cout<<"\n error_"<<i<<" "<<err;
		
		//for(int j=1;j<=count;++j)
		//	X[j] -= yc; 
			
		//std::cout<<"\n X_"<<i<<" "<<X;
		
		
/// x_i^TP
		xtp.setZero();
		P.lefthandMult(xtp, X);
	
		//std::cout<<"\n X^tP"<<xtp; 
		float scal = 1.f / (lamda + xtp.dot(X) );

/// Q_i <- P x_i / (lamda + x_i^T P x_i) 	
		Q.setZero();
		P.mult(Q, X);
		Q.scale(scal);
	
		//std::cout<<"\n Q"<<Q;
/// beta <- beta + Q_i error_i
		beta.add(Q * err);
		//std::cout<<"\n b_"<<i<<beta;
		
/// current estimate
/// y_i^hat <- beta_i^T x_i
		float y_hat = beta.dot(X);
		std::cout<<"\n y_"<<i<<" "<<Y[i]<<" y_"<<i<<"hat "<<y_hat;	
	
/// P_i <- (P_i - Q_i x_i^T P_i) / lamda
		qxt.setZero();
		qxt.asVVt(Q, X);
		//std::cout<<"\n Qx^t "<<qxt;
	
		qxtp.setZero();
		qxt.mult(qxtp, P);
		//std::cout<<"\n Qx^tP "<<qxtp;
	
		P.minus(qxtp);
		P.scale(1.f/lamda);
		//std::cout<<"\n P_"<<i<<P;
	}

}

void testPredictor()
{
	LinearRegressionData<float, 4> model;
	LinearRegressionPredictor<float, 4> estimator;
	estimator.setData(&model);
	
	const int n = 25;
	DenseVector<float> Y(n);
	for(int i=0;i<n;++i) {
		Y[i] = .5f + .02f * i + RandomFn11() * .02f;
	}
	
	for(int t=0;t<n;++t) {
		float yhat = estimator.updateAndPredict(Y[t], t);
		std::cout<<"\n t="<<t<<" input signal "<<Y[t]<<" estimate "<<yhat;
	}
}

int main(int argc, char **argv)
{        
    std::cout<<"\n test linear model";
	//testPredictor();
	//testEstimation();
	//testVVt();
	std::cout<<"\ndone.\n";
    //exit(0);
    QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
    window.resize(480, 480);
    window.show();
    return app.exec();
}
