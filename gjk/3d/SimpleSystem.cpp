/*
 *  SimpleSystem.cpp
 *  proof
 *
 *  Created by jian zhang on 1/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "SimpleSystem.h"
#define IDIM  10
#define IDIM1 11
#define timeStep 0.0166667f
#ifdef DBG_DRAW
#include <KdTreeDrawer.h>
#endif
SimpleSystem::SimpleSystem()
{
	m_groundX = new Vector3F[IDIM1 * IDIM1];
	m_groundIndices = new unsigned[numGroundFaceVertices()];
	
	unsigned *ind = &m_groundIndices[0];
	unsigned i, j, i1, j1;
	for(j=0; j < IDIM; j++) {
	    j1 = j + 1;
		for(i=0; i < IDIM; i++) {
		    i1 = i + 1;
			*ind = j * IDIM1 + i;
			ind++;
			*ind = j1 * IDIM1 + i;
			ind++;
			*ind = j * IDIM1 + i1;
			ind++;

			*ind = j * IDIM1 + i1;
			ind++;
			*ind = j1 * IDIM1 + i;
			ind++;
			*ind = j1 * IDIM1 + i1;
			ind++;
		}
	}
	
	Vector3F * v = &m_groundX[0];
	for(j=0; j < IDIM1; j++) {
	    for(i=0; i < IDIM1; i++) {
		    i1 = i + 1;
			v->set(i * 3.f, -9.f, j * 3.f);
			v++;
		}
	}
	
	m_X = new Vector3F[3];
	m_X[0].set(1.1f, 1.3f, 10.8f);
	m_X[1].set(2.6f, 1.7f, 10.1f);
	m_X[2].set(2.5f, 4.3f, 10.4f);
	
	m_indices = new unsigned[3];
	m_indices[0] = 0;
	m_indices[1] = 1;
	m_indices[2] = 2;
	
	m_V = new Vector3F[3];
	m_Vline = new Vector3F[3 * 2];
	m_vIndices = new unsigned[3 * 2];
	
	for(i = 0; i< 3; i++) {
		m_V[i].set(10.f, 0.f, 0.f);
		m_Vline[i*2] = m_X[i];
		m_Vline[i*2 + 1] = m_X[i] + m_V[i] * timeStep;
		m_vIndices[i*2] = i*2;
		m_vIndices[i*2+1] = i*2+1;
	}
	
	m_rb.position.set(-10.f, 17.f, 15.f);
	m_rb.orientation.set(1.f, 0.f, 0.f, 0.f);
	m_rb.linearVelocity.set(2.f, 0.f, 0.f);
	m_rb.angularVelocity.setZero();
	m_rb.shape = new CuboidShape(4.f, 4.f, 4.f);
	m_rb.shape->setMass(1.f);
	
	m_ground.position.set(-15.f, -7.f, 15.f);
	m_ground.orientation.set(1.f, 0.f, 0.f, 0.f);
	m_ground.linearVelocity.setZero();
	m_ground.angularVelocity.setZero();
	TetrahedronShape * tet = new TetrahedronShape;
	tet->p[0].set(-10.f, 10.f, -20.f);
	tet->p[1].set(-10.f, 10.f, 120.f);
	tet->p[2].set(80.f, 10.f, -20.f);
	tet->p[3].set(0.f, -20.f, -20.f);
	m_ground.shape = tet;
	m_ground.shape->setMass(10.f);
}

Vector3F * SimpleSystem::groundX() const
{ return m_groundX; }

const unsigned SimpleSystem::numGroundFaceVertices() const
{ return IDIM * IDIM * 2 * 3; }

unsigned * SimpleSystem::groundIndices() const
{ return m_groundIndices; }

Vector3F * SimpleSystem::X() const
{ return m_X; }

const unsigned SimpleSystem::numFaceVertices() const
{ return 3; }

unsigned * SimpleSystem::indices() const
{ return m_indices; }

Vector3F * SimpleSystem::Vline() const
{ return m_Vline; }

const unsigned SimpleSystem::numVlineVertices() const
{ return 3 * 2; }

unsigned * SimpleSystem::vlineIndices() const
{ return m_vIndices; }

void SimpleSystem::progress()
{
	int i;
	for(i = 0; i< 3; i++) {
		m_V[i] += Vector3F(0.f, -980.f, 0.f) * timeStep;
	}
	
	for(i = 0; i< 3; i++) {
		m_X[i] += m_V[i] * timeStep;
	}
	
	for(i = 0; i< 3; i++) {
		m_Vline[i*2] = m_X[i];
		m_Vline[i*2 + 1] = m_X[i] + m_V[i] * timeStep;
	}
	
	applyGravity();
	applyImpulse();
	applyVelocity();
	updateStates();
}

RigidBody * SimpleSystem::rb()
{ return &m_rb; }

RigidBody * SimpleSystem::ground()
{ return &m_ground; }

void SimpleSystem::applyGravity()
{ 
	m_rb.linearVelocity += Vector3F(0.f, -9.8f, 0.f) * timeStep;
}

void SimpleSystem::applyImpulse()
{
    /*
    m_rb.linearVelocity.set(0.f, -9.8/30.f, 0.f);
    
    Vector3F linearJ(0, 1, 0);
    
    Matrix33F R; R.set(m_rb.orientation); R.inverse();
    Vector3F ro = R.transform(Vector3F(0, 1, 0));
    
    Vector3F angularJ = Vector3F(-4, -4, -4).cross(ro).reversed(); 

    Vector3F linearM = m_rb.shape->linearMassM;

	Matrix33F angularM = m_rb.shape->angularMassM;

    Vector3F linearJMinv(linearJ.x * linearM.x, linearJ.y * linearM.y, linearJ.z * linearM.z);
    
	Vector3F angularJMinv = angularM.transform(angularJ);
    
    float JMinvJt = linearJMinv.dot(linearJ) + angularJMinv.dot(angularJ);
    
    float Jv = linearJ.dot(m_rb.linearVelocity) + angularJ.dot(m_rb.angularVelocity);
    
    float lamda = - Jv / JMinvJt;
    Vector3F linearMinvJt(linearM.x * linearJ.x, linearM.y * linearJ.y, linearM.z * linearJ.z);
	Vector3F angularMinvJt = angularM * angularJ;

    if(Jv != 0.f) {
        	angularJ.verbose("angular J");
    linearM.verbose("linear M");
    std::cout<<"angular M"<<angularM.str()<<"\n";
	linearJMinv.verbose("linearJMinv");
    angularJMinv.verbose("angular JMinv");
    std::cout<<"JMinvJt"<<JMinvJt<<"\n";
    std::cout<<"Jv "<<linearJ.dot(m_rb.linearVelocity)<<" + "<<angularJ.dot(m_rb.angularVelocity)<<"="<<Jv<<"\n";
    angularMinvJt.verbose("angularMinvJt");
    }
    
    m_rb.linearVelocity += linearMinvJt * lamda;
    m_rb.angularVelocity += angularMinvJt * lamda;
    */
	m_rb.TOI = 1.f;
	continuousCollisionDetection(m_ground, m_rb);
	if(!m_ccd.hasContact) {
		return;
	}
	if(m_ccd.TOI == 0.f) {
		// m_rb.linearVelocity.setZero();
		// m_rb.angularVelocity.setZero();
		// m_rb.projectedLinearVelocity.setZero();
		// m_rb.projectedAngularVelocity.setZero();
		// return;
	}
	
	if(m_ccd.TOI > 0.f) std::cout<<" toi "<<m_ccd.TOI<<"\n";
	
	m_rb.TOI = m_ccd.TOI;

// B at impact position
	m_ccd.positionB = m_rb.position.progress(m_rb.linearVelocity, m_rb.TOI * timeStep);
	m_ccd.orientationB = m_rb.orientation.progress(m_rb.angularVelocity, m_rb.TOI * timeStep);
	
	m_rb.projectedLinearVelocity = m_rb.linearVelocity;
	m_rb.projectedAngularVelocity = m_rb.angularVelocity;
	
	float lastLamda;
	float lamda = 0.f;
	for(int i=0; i<4; i++) {
		if(!m_ccd.hasContact) {
			// std::cout<<" no contact this iteration\n";
			continue;
		}
		// std::cout<<"contact n"<<m_ccd.contactNormal.str();
		
		lastLamda = lamda;
		
		Vector3F linearJ;
		Vector3F angularJ;
	
		linearJ = m_ccd.contactNormal;

		Matrix33F R; R.set(m_ccd.orientationB/*.progress(m_rb.projectedAngularVelocity, timeStep)*/); R.inverse();
		
		// std::cout<<"o("<<m_rb.orientation.x<<","<<m_rb.orientation.y<<","<<m_rb.orientation.z<<")"; 
		// std::cout<<"R"<<R.str();
		
		Vector3F ro = R.transform(m_ccd.contactNormal);
		ro.normalize(); 
		// std::cout<<"||r||"<<ro.str();
		angularJ = m_ccd.contactPointB.reversed().cross(ro);
		// std::cout<<"contact p"<<m_ccd.contactPointB.str();
		// std::cout<<"linearJ"<<linearJ.str();
		// std::cout<<"angularJ"<<angularJ.str();
	
#ifdef DBG_DRAW
    KdTreeDrawer * drawer = m_gjk.m_dbgDrawer;
    Matrix44F space;
	space.setRotation(m_rb.orientation);
	space.setTranslation(m_rb.position);
    
    Vector3F wb = space.transform(m_ccd.contactPointB);
	glBegin(GL_LINES);
	glColor3f(0.f, 1.f, 1.f);
	glVertex3f(wb.x, wb.y, wb.z);
	glVertex3f(wb.x + linearJ.x, wb.y + linearJ.y, wb.z + linearJ.z);
	glEnd();
	
	if(m_ccd.TOI == 0.f) {
		glColor3f(1.f, 0.f, 0.f);
		drawer->circleAt(wb, m_ccd.contactNormal);
	}
    
	glPushMatrix();
	drawer->useSpace(space);
	glBegin(GL_LINES);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(m_ccd.contactPointB.x, m_ccd.contactPointB.y, m_ccd.contactPointB.z);
	glVertex3f(0.f, 0.f, 0.f);
	glColor3f(1.f, 1.f, 0.f);
	glVertex3f(m_ccd.contactPointB.x, m_ccd.contactPointB.y, m_ccd.contactPointB.z);
	glVertex3f(m_ccd.contactPointB.x + linearJ.x, m_ccd.contactPointB.y + linearJ.y, m_ccd.contactPointB.z + linearJ.z);
	glColor3f(0.f, 1.f, .3f);
	glVertex3f(m_ccd.contactPointB.x, m_ccd.contactPointB.y, m_ccd.contactPointB.z);
	glVertex3f(m_ccd.contactPointB.x + angularJ.x, m_ccd.contactPointB.y + angularJ.y, m_ccd.contactPointB.z + angularJ.z);
	glEnd();
	glPopMatrix();
#endif

		Vector3F linearM = m_rb.shape->linearMassM; 
		Matrix33F angularM = m_rb.shape->angularMassM;	
		Vector3F linearJMinv(linearJ.x * linearM.x, linearJ.y * linearM.y, linearJ.z * linearM.z);
		Vector3F angularJMinv = angularM.transform(angularJ);
	
		float JMinvJt = linearJMinv.dot(linearJ) + angularJMinv.dot(angularJ);
	
		// std::cout<<" JMinvJt "<<JMinvJt<<"\n";
		if(JMinvJt < TINY_VALUE) continue;
					
		float Jv = linearJ.dot(m_rb.projectedLinearVelocity) + angularJ.dot(m_rb.projectedAngularVelocity);
	
		// std::cout<<" Jv l "<<linearJ.dot(m_rb.linearVelocity)<<" a "<<angularJ.dot(m_rb.angularVelocity);
		// std::cout<<" Jv "<<Jv;
	
		lamda = - Jv / JMinvJt;
	
		char showStop = 0;
		// std::cout<<"\n impulse "<<lamda<<"\n";
		if(lamda< 0.f) {
			lamda = 0.f;
			showStop = 1;
		}
	
		Vector3F linearMinvJt(linearM.x * linearJ.x, linearM.y * linearJ.y, linearM.z * linearJ.z);
		//linearMinvJt.verbose("linearMinvJt");
		Vector3F angularMinvJt = angularM * angularJ;
		// angularJ.verbose("angularJ");
		// angularMinvJt.verbose("angularMinvJt");
		
// http://www.cs.uu.nl/docs/vakken/mgp/lectures/lecture%207%20Collision%20Resolution.pdf		
		float Jb = -(1.f + .49f) * m_rb.projectedLinearVelocity.reversed().dot(m_ccd.contactNormal);
		
		if(m_ccd.penetrateDepth > 0.f) {
			// std::cout<<" penetrate d "<<m_ccd.penetrateDepth;
			Jb += m_ccd.penetrateDepth * 60.f;
		}
		
		Vector3F bigI = m_ccd.contactNormal * -Jb;
#ifdef DBG_DRAW
	glBegin(GL_LINES);
	glColor3f(1,1,1);
	glVertex3f(m_rb.position.x, m_rb.position.y, m_rb.position.z);
	glVertex3f(m_rb.position.x + bigI.x, m_rb.position.y + bigI.y, m_rb.position.z + bigI.z);
	glEnd();
#endif		
		m_rb.projectedLinearVelocity += bigI;
		// m_rb.projectedAngularVelocity += angularMinvJt * (lamda - lastLamda);
		
		//m_rb.projectedLinearVelocity.verbose("pv");
		// m_rb.projectedAngularVelocity.verbose("pav");

#ifdef DBG_DRAW
	glBegin(GL_LINES);
	glColor3f(1,1,1);
	glVertex3f(m_rb.position.x, m_rb.position.y, m_rb.position.z);
	glVertex3f(m_rb.position.x + linearMinvJt.x * (lamda - lastLamda), m_rb.position.y + linearMinvJt.y * (lamda - lastLamda), m_rb.position.z + linearMinvJt.z * (lamda - lastLamda));
	glEnd();
	
	glPushMatrix();
	drawer->useSpace(space);
	glColor3f(0,1,1);
	drawer->arrow(Vector3F::Zero, angularMinvJt);
	glPopMatrix();
#endif
		
// collision at impact position
		m_ccd.linearVelocityB = m_rb.projectedLinearVelocity * timeStep;
		m_ccd.angularVelocityB = m_rb.projectedAngularVelocity * timeStep;
		// std::cout<<" test at impact point ";
		m_gjk.timeOfImpact(*m_ground.shape, *m_rb.shape, &m_ccd);
	}
	// m_rb.linearVelocity.verbose("v");
	// m_rb.angularVelocity.verbose("av");
	// m_rb.projectedLinearVelocity.verbose("pv");
	// m_rb.projectedAngularVelocity.verbose("pav");
	
	//m_rb.projectedLinearVelocity.set(0, 100, 0);
	//m_rb.linearVelocity = m_rb.projectedLinearVelocity;
	//m_rb.angularVelocity = m_rb.projectedAngularVelocity;
}

void SimpleSystem::continuousCollisionDetection(const RigidBody & A, const RigidBody & B)
{
	ContinuousCollisionContext &io = m_ccd;
	io.positionA = A.position;
	io.positionB = B.position;
	io.orientationA = A.orientation;
	io.orientationB = B.orientation;
	io.linearVelocityA = A.linearVelocity * timeStep;
	io.linearVelocityB = B.linearVelocity * timeStep;
	io.angularVelocityA = A.angularVelocity * timeStep;
	io.angularVelocityB = B.angularVelocity * timeStep;
	m_gjk.timeOfImpact(*A.shape, *B.shape, &m_ccd);
}

void SimpleSystem::applyVelocity()
{
	m_rb.integrateP(timeStep);
}
#ifdef DBG_DRAW
void SimpleSystem::setDrawer(KdTreeDrawer * d)
{
	m_gjk.m_dbgDrawer = d;
}
#endif

void SimpleSystem::drawWorld()
{
#ifdef DBG_DRAW
    KdTreeDrawer * drawer = m_gjk.m_dbgDrawer;
    Matrix33F mat;
    
    Vector3F at = m_rb.position;
	mat.set(m_rb.orientation);
	drawer->coordsys(mat, 8.f, &at);
	CuboidShape * cub = static_cast<CuboidShape *>(m_rb.shape);
	
	glColor3f(0.f, 0.5f, 0.f);
	glPushMatrix();
	Matrix44F space;
	space.setRotation(mat);
	space.setTranslation(at);
	drawer->useSpace(space);
	drawer->aabb(Vector3F(-cub->m_w, -cub->m_h, -cub->m_d), Vector3F(cub->m_w, cub->m_h, cub->m_d));
	glPopMatrix();
	
	Vector3F vel = m_rb.linearVelocity;
	glColor3f(0.f, 0.f, .5f);
	drawer->arrow(at, at + vel);
	
	vel = m_rb.angularVelocity;
	glColor3f(0.f, 0.5f, 0.f);
	drawer->arrow(at, at + vel);
	
	at = m_ground.position;
	mat.set(m_ground.orientation);
	TetrahedronShape * tet = static_cast<TetrahedronShape *>(m_ground.shape);
	
	glColor3f(0.f, 0.f, 0.5f);
	glPushMatrix();
	space.setRotation(mat);
	space.setTranslation(at);
	drawer->useSpace(space);
	drawer->tetrahedron(tet->p);
	glPopMatrix();
#endif
}

void SimpleSystem::updateStates()
{
	m_rb.updateState();
}
