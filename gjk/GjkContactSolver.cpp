/*
 *  GjkContactSolver.cpp
 *  proof
 *
 *  Created by jian zhang on 1/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "GjkContactSolver.h"

GjkContactSolver::GjkContactSolver() {}

void GjkContactSolver::separateDistance(const PointSet & A, const PointSet & B, ClosestTestContext * result)
{
	resetSimplex(result->W);
	float v2;
	Vector3F w, pa, pb;
	Vector3F v = result->transformA.transform(A.X[0]);
	if(v.length2() < TINY_VALUE) v = result->transformA.transform(A.X[1]);
	Vector3F localA, localB;
	
	for(int i=0; i < 99; i++) {
		// SA-B(-v)
	    pa = A.supportPoint(v.reversed(), result->transformA, localA);
		pb = B.supportPoint(v, result->transformB, localB);
		w = pa - pb;// + v.normal() * MARGIN_DISTANCE;
	    
		// terminate when v is close enough to v(A - B).
	    // http://www.bulletphysics.com/ftp/pub/test/physics/papers/jgt04raycast.pdf
		v2 = v.length2();
	    if(v2 - w.dot(v) < 0.0001f * v2) {
	        // std::cout<<" v is close to w "<<v2 - w.dot(v)<<"\n";
			result->hasResult = 0;
			break;
	    }
	    
	    addToSimplex(result->W, w, localB);
 
	    if(isPointInsideSimplex(result->W, result->referencePoint)) {
	        // std::cout<<" Minkowski difference contains the reference point\n";
			result->hasResult = 1;
			break;
	    }
	    
	    result->hasResult = 0;
		result->distance = 1e9;
		closestOnSimplex(result);
	    v = result->closestPoint - result->referencePoint;
		result->separateAxis = v;
		interpolatePointB(result);
		// in world space
		smallestSimplex(result);
	}
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
		pa = A.supportPoint(v, result->transformA, localA);
		pb = B.supportPoint(v.reversed(), result->transformB, localB);
		p = pa - pb;
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
	
	result->hasResult = 1;
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
		pa = A.supportPoint(v, result->transformA, localA);
		pb = B.supportPoint(v.reversed(), result->transformB, localB);
	    p = pa - pb;// + v.normal() * MARGIN_DISTANCE;
	    
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
    const Vector3F relativeLinearVelocity = result->linearVelocityB - result->linearVelocityA;
    const float angularMotionSize = result->angularVelocityA.length() * A.angularMotionDisc()
                                    + result->angularVelocityB.length() * B.angularMotionDisc();
    // no relative motion
    if(relativeLinearVelocity.length() + angularMotionSize < TINY_VALUE)
        return;
    
    ClosestTestContext separateIo;
	separateIo.transformA.setTranslation(result->positionA);
	separateIo.transformA.setRotation(result->orientationA);
    separateIo.transformB.setTranslation(result->positionB);
	separateIo.transformB.setRotation(result->orientationB);
    separateIo.referencePoint.setZero();
	separateIo.needContributes = 1;
	separateIo.distance = 1e9;
	separateIo.rayDirection = relativeLinearVelocity.normal();
	
	separateDistance(A, B, &separateIo);
    
    result->TOI = 0.f;
    
    // already contacted
    if(separateIo.hasResult) {
		std::cout<<" contacted at t0\n";
        return;
	}
    
    float distance = separateIo.separateAxis.length();
    Vector3F separateN = separateIo.separateAxis / distance;
    
    float closeInSpeed = relativeLinearVelocity.dot(separateN);
    
    // going apart
    if(closeInSpeed + angularMotionSize < TINY_VALUE) {
		std::cout<<" going away\n";
        return;
	}
    
    const Vector3F position0A = result->positionA;
    const Vector3F position0B = result->positionB;
    const Quaternion orientation0A = result->orientationA;
    const Quaternion orientation0B = result->orientationB;
    
    float lamda = 0.f;
	float lastLamda = 0.f;

    int k = 0;
    for(; k < 32; k++) {
		lamda += distance / (closeInSpeed + angularMotionSize);
		//std::cout<<" "<<k<<" lamda "<<lamda<<" dist "<<distance<<"\n";
        
        if(lamda < 0.f) {
			// std::cout<<"lamda < 0\n";
			return;
		}
        if(lamda > 1.f) {
			// std::cout<<"lamda > 1\n";
			return;
		}
        
        separateIo.transformA.setTranslation(position0A.progress(result->linearVelocityA, lamda));
        separateIo.transformA.setRotation(orientation0A.progress(result->angularVelocityA, lamda));
        separateIo.transformB.setTranslation(position0B.progress(result->linearVelocityB, lamda));
        separateIo.transformB.setRotation(orientation0B.progress(result->angularVelocityB, lamda));
        separateIo.referencePoint.setZero();
		separateIo.distance = 1e9;
		separateDistance(A, B, &separateIo);
        
        if(separateIo.hasResult) {
			if(separateIo.distance < 0.001f) break;
			std::cout<<" "<<k<<" contacted "<<lamda<<"\n";
			lamda = lastLamda;
			//std::cout<<" last "<<lastLamda<<" back "<<lamda<<"\n";
			
			// std::cout<<" move back d "<<distance<<" - "<<separateIo.distance<<" ";
			distance -= separateIo.distance * relativeLinearVelocity.normal().dot(separateN);
			// std::cout<<" = "<<distance<<"\n";

			continue;
		}
		
		distance = separateIo.separateAxis.length();
		
        if(distance < 0.001f) {
			std::cout<<" "<<k<<" close enough at "<<lamda<<"\n";
			break;
		}
		
		separateN = separateIo.separateAxis / distance;
		
		//separateIo.rayDirection = separateN;
				
		// std::cout<<" "<<k<<" sep "<<lamda<<" : "<<distance<<"\n";

        closeInSpeed = relativeLinearVelocity.dot(separateN);
		
		// std::cout<<" close "<<closeInSpeed<<"\n";
    
        if(closeInSpeed + angularMotionSize < TINY_VALUE) {
			// std::cout<<"go apart at time "<<lamda<<"\n";
            return;
		}
		
		// keep the separated time
		lastLamda = lamda;

    }
	if(k==32) {
		std::cout<<"max n iteration reached!\n";
		std::cout<<" "<<k<<" "<<lamda<<" : "<<distance<<"\n";
	}
	result->TOI = lamda;
	result->contactNormal = separateN;
	result->contactPointB = separateIo.contactPointB;
		
}
