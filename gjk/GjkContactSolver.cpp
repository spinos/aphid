/*
 *  GjkContactSolver.cpp
 *  proof
 *
 *  Created by jian zhang on 1/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "GjkContactSolver.h"
#ifdef DBG_DRAW
#include <KdTreeDrawer.h>
#endif
GjkContactSolver::GjkContactSolver() {}

void GjkContactSolver::separateDistance(const PointSet & A, const PointSet & B, ClosestTestContext * result)
{
	resetSimplex(result->W);
	float v2;
	Vector3F w, pa, pb;
	Vector3F v = result->transformA.transform(A.X[0]) - result->referencePoint;
	if(v.length2() < TINY_VALUE) v = result->transformA.transform(A.X[1]) - result->referencePoint;
	Vector3F localA, localB;
	
	for(int i=0; i < 99; i++) {
		// SA-B(-v)
	    pa = A.supportPoint(v.reversed(), result->transformA, localA, result->margin);
		pb = B.supportPoint(v, result->transformB, localB, result->margin);
		w = pa - pb;
	    
		// terminate when v is close enough to v(A - B).
	    // http://www.bulletphysics.com/ftp/pub/test/physics/papers/jgt04raycast.pdf
		v2 = v.length2();
	    if(v2 - w.dot(v) < 0.0001f * v2) {
	        // std::cout<<" v is close to w "<<v2 - w.dot(v)<<"\n";
			break;
	    }
	    
	    addToSimplex(result->W, w, localB);
		
#ifdef DBG_DRAW
		glPushMatrix();
		m_dbgDrawer->useSpace(result->transformB);
		glColor3f(0.f, .5f, 0.f);
		m_dbgDrawer->arrow(Vector3F::Zero, localB);
		glPopMatrix();
		
		glPushMatrix();
		m_dbgDrawer->useSpace(result->transformA);
		glColor3f(.5f, 0.f, 0.f);
		m_dbgDrawer->arrow(Vector3F::Zero, localA);
		glPopMatrix();
		
		glPushMatrix();
		m_dbgDrawer->useSpace(result->transformA);
		glColor3f(.5f, .5f, 0.f);
		m_dbgDrawer->arrow(Vector3F::Zero, w + result->transformB.transform(localB));
		glPopMatrix();
#endif
 
	    if(isPointInsideSimplex(result->W, result->referencePoint)) {
	        // std::cout<<" Minkowski difference contains the reference point\n";
			result->hasResult = 1;
			return;
	    }
	    
	    result->hasResult = 0;
		result->distance = 1e9;
		closestOnSimplex(result);
	    v = result->closestPoint - result->referencePoint;
		result->separateAxis = v;
#ifdef DBG_DRAW
		glColor3f(.1f, 3.f, 0.f);
		m_dbgDrawer->arrow(Vector3F::Zero, v);
#endif
		interpolatePointB(result);
		// in world space
		smallestSimplex(result);
	}
	result->hasResult = 0;
}

void GjkContactSolver::penetration(const PointSet & A, const PointSet & B, ClosestTestContext * result)
{
	resetSimplex(result->W);
	const Vector3F r = result->rayDirection;
	const Vector3F startP = Vector3F::Zero - result->rayDirection * 99.f;
	Vector3F hitP = startP;
	// from origin to startP
	Vector3F v = hitP;
	Vector3F w, p, pa, pb, localA, localB;
	float lamda = 0.f;
	float vdotw, vdotr;

	int k = 0;
	for(; k < 39; k++) {
		vdotr = v.dot(r);
	
		// SA-B(v)
		pa = A.supportPoint(v, result->transformA, localA, result->margin);
		pb = B.supportPoint(v.reversed(), result->transformB, localB, result->margin);
		p = pa - pb;// + v.normal() * MARGIN_DISTANCE;
		w = hitP - p;
		vdotw = v.dot(w); 
		
		if(vdotw > 0.f) {
			if(vdotr >= 0.f)
				break;
			lamda -= vdotw / vdotr;
			hitP = startP + r * lamda;
		}
				
		addToSimplex(result->W, p, localB);
	
		result->hasResult = 0;
		result->distance = 1e9;
		result->referencePoint = hitP;
	
		closestOnSimplex(result);
	
		v = hitP - result->closestPoint;
		
		interpolatePointB(result);
	
		if(v.length2() < TINY_VALUE) break;
		
		result->separateAxis = v;
	
		smallestSimplex(result);
	}
	
	result->distance = hitP.length();
	result->separateAxis.normalize();
}

void GjkContactSolver::rayCast(const PointSet & A, const PointSet & B, ClosestTestContext * result)
{
	separateDistance(A, B, result);
	if(result->hasResult) return;
	
	resetSimplex(result->W);
	
	const Vector3F r = result->rayDirection;
	float lamda = 0.f;
	// ray started at origin
	const Vector3F startP = Vector3F::Zero;
	Vector3F hitP = startP;
	Vector3F hitN; hitN.setZero();
	Vector3F v = hitP - result->closestPoint;
	Vector3F w, p, pa, pb, localA, localB;
	
	float vdotw, vdotr;
	int k = 0;
	for(; k < 32; k++) {
	    vdotr = v.dot(r);
	    
	    // SA-B(v)
		pa = A.supportPoint(v, result->transformA, localA, result->margin);
		pb = B.supportPoint(v.reversed(), result->transformB, localB, result->margin);
	    p = pa - pb;
	    
		w = hitP - p;
	    vdotw = v.dot(w); 
	    
	    if(vdotw > 0.f) {
			// std::cout<<" v.w > 0\n";
			if(vdotr >= 0.f) {
				// std::cout<<" v.r >= 0 missed\n";
				result->hasResult = 0;
				return;
			}
			lamda -= vdotw / vdotr;
			hitP = startP + r * lamda;
			hitN = v;
		}
		
	    addToSimplex(result->W, p, localB);
	    
	    result->hasResult = 0;
	    result->distance = 1e9;
	    result->referencePoint = hitP;
	    
	    closestOnSimplex(result);
	    
	    v = hitP - result->closestPoint;
		
		interpolatePointB(result);
		
		if(v.length2() < TINY_VALUE) break;
		
		smallestSimplex(result);
	}
	
	if(k==32) std::cout<<"    max iterations reached!\n";
	// std::cout<<" k"<<k<<" ||v|| "<<v.length()<<"\n";
	result->hasResult = 1;
	result->separateAxis = hitN.normal();
	result->distance = lamda;
}

void GjkContactSolver::timeOfImpact(const PointSet & A, const PointSet & B, ContinuousCollisionContext * result)
{
    result->hasContact = 0;
    result->penetrateDepth = 0.f;
    result->TOI = 0.f;
    
    const Vector3F relativeLinearVelocity = result->linearVelocityB - result->linearVelocityA;
    
    // std::cout<<" velocityA "<<result->linearVelocityA.str();
	// std::cout<<" velocityB "<<result->linearVelocityB.str();
    // std::cout<<" relativeLinearVelocity "<<relativeLinearVelocity.str();
		
    const float angularMotionSize = result->angularVelocityA.length() * A.angularMotionDisc()
                                    + result->angularVelocityB.length() * B.angularMotionDisc();
    // no relative motion
    if(relativeLinearVelocity.length() + angularMotionSize < TINY_VALUE)
		return;
		
#ifdef DBG_DRAW		
	Vector3F lineB = result->positionB;
	Vector3F lineE = lineB + relativeLinearVelocity;
	glColor3f(0.f, .1f, .6f);
	m_dbgDrawer->arrow(lineB, lineE);
	lineB = result->positionA;
	lineE = lineB - relativeLinearVelocity;
	glColor3f(0.f, .1f, .6f);
	m_dbgDrawer->arrow(lineB, lineE);
#endif
		
    ClosestTestContext separateIo;
	separateIo.needContributes = 1;
	separateIo.margin = 0.05f;
    
    Vector3F separateN;
    
    float distance, closeInSpeed;
	float lastDistance = 0.f;
	float dDistanceaLamda;
    
    const Vector3F position0A = result->positionA;
    const Vector3F position0B = result->positionB;
    const Quaternion orientation0A = result->orientationA;
    const Quaternion orientation0B = result->orientationB;
    
    float lamda = 0.f;
	float limitDeltaLamda, deltaLamda = 1.f;
	float lastLamda = 0.f;
	
	int k = 0;
    for(; k < 32; k++) {
        
        separateIo.transformA.setTranslation(position0A.progress(result->linearVelocityA, lamda));
		Quaternion ra = orientation0A.progress(result->angularVelocityA, lamda);
		//ra.normalize();
        separateIo.transformA.setRotation(ra);
        separateIo.transformB.setTranslation(position0B.progress(result->linearVelocityB, lamda));
		Quaternion rb = orientation0B.progress(result->angularVelocityB, lamda);
		//rb.normalize();
        separateIo.transformB.setRotation(rb);
        separateIo.referencePoint.setZero();
		separateIo.distance = 1e9;
		separateDistance(A, B, &separateIo);
        
        if(separateIo.hasResult) {
            if(k<1) {	
				// std::cout<<"     contact at t0 try zero margin\n";
                separateIo.margin = 0.f;
				separateIo.distance = 1e9;
				separateDistance(A, B, &separateIo);
				if(separateIo.hasResult) {
					std::cout<<"     intersected\n";
					result->hasContact = 0;
					return;
				}
				
				result->contactPointB = separateIo.contactPointB;
				distance = separateIo.separateAxis.length();
				result->penetrateDepth = 0.1 - distance;
				separateN = separateIo.separateAxis / -distance;
#ifdef DBG_DRAW		
		lineB = separateIo.transformB.transform(separateIo.contactPointB);
		lineE = lineB + separateN;
		glColor3f(1.f, 0.f, 0.f);
		m_dbgDrawer->arrow(lineB, lineE);
#endif
				break;
            } else {
				// std::cout<<" contact at "<<lamda;;
				lamda = lastLamda;
				break;
			} 
		}
		
		result->contactPointB = separateIo.contactPointB;
		
		distance = separateIo.separateAxis.length();
				
        if(distance < .001f) {
			// std::cout<<" "<<k<<" close enough at "<<lamda<<"\n";
			if(k<1) {
				separateIo.margin = 0.f;
				separateIo.distance = 1e9;
				separateDistance(A, B, &separateIo);
				result->contactPointB = separateIo.contactPointB;
				distance = separateIo.separateAxis.length();
				separateN = separateIo.separateAxis / -distance;
			}
			break;
		}
		
		separateN = separateIo.separateAxis / distance;
		
#ifdef DBG_DRAW		
		lineB = separateIo.transformB.transform(separateIo.contactPointB);
		lineE = lineB + separateN;
		glColor3f(1.f, 0.f, 0.f);
		m_dbgDrawer->arrow(lineB, lineE);
#endif

		dDistanceaLamda = (distance - lastDistance) / deltaLamda;
		lastDistance = distance;
		
		// std::cout<<" sep ax "<<separateIo.separateAxis.str();
		// std::cout<<" dist "<<distance;
		// std::cout<<" sep n "<<separateN.str();
				
		closeInSpeed = relativeLinearVelocity.dot(separateN);
		// std::cout<<" closeInSpeed "<<closeInSpeed;

        if(closeInSpeed + angularMotionSize < 0.f) {
			// std::cout<<"go apart at time "<<lamda<<"\n";
			return;
		}
		
		deltaLamda = distance / (closeInSpeed + angularMotionSize);
		if(dDistanceaLamda < 0.f) {
			limitDeltaLamda = -.59f * lastDistance / dDistanceaLamda;
			if(deltaLamda > limitDeltaLamda) {
				deltaLamda = limitDeltaLamda;
				// std::cout<<" limit delta lamda "<<deltaLamda<<"\n";
			}
		}
		lastLamda = lamda;
		lamda += deltaLamda;

        if(lamda < 0.f) {
			// std::cout<<"lamda < 0\n";
			return;
		}
        if(lamda > 1.f) {
			// std::cout<<"lamda > 1\n";
			return;
		}
    }
	
    result->hasContact = 1;
	result->TOI = lamda;
	result->contactNormal = separateN.reversed();
#ifdef DBG_DRAW		
	lineB = separateIo.transformB.transform(separateIo.contactPointB);
	lineE = lineB + result->contactNormal;
	glColor3f(.2f, 1.f, .1f);
	m_dbgDrawer->arrow(lineB, lineE);
#endif
}
