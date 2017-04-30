#include <QtCore>
#include "SolverThread.h"
#include <BoxProgram.h>
#define NU 99
#define NV 99
#define NF (NU * NV)
#define NTri (NF * 2)
#define NI (NTri * 3)
#define NP ((NU + 1) * (NV + 1))

#define STRUCTURAL_SPRING 0
#define SHEAR_SPRING 1
#define BEND_SPRING 2
#define EPSILON  0.0000001f

const float DEFAULT_DAMPING =  -0.0825f;
float	KsStruct = 180.5f,KdStruct = -.25f;
float	KsShear = 180.5f,KdShear = -.25f;
float	KsBend = 80.5f,KdBend = -.25f;
float timeStep = 1.f / 720.f;
Vector3F gravity(0.0f,-981.f,0.0f);
float mass = 1.f;
float iniHeight = 99.f;
float gridSize = iniHeight / (float)NU;
const unsigned solver_iterations = 2;
float kBend = 0.5f;
float kStretch = 1.f; 
float kDamp = 0.0125f;
float global_dampening = 0.99f;

SolverThread::SolverThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;
    
    m_pos = new Vector3F[NP];
	m_posLast = new Vector3F[NP];
	m_force = new Vector3F[NP];
	m_indices = new unsigned[NI];
	m_invMass = new float[NP];
	m_velocity = new Vector3F[NP];
	m_Ri = new Vector3F[NP];
	m_projectedPos = new Vector3F[NP];
	
	unsigned i, j;
	for(i=0; i < NP; i++) m_invMass[i] = 1.f / mass;
	
	// for(i=0; i < NU; i++) m_invMass[i] = 0.f;
	m_invMass[0] = 0.f;
	m_invMass[NU] = 0.f;

	for(i=0; i < NP; i++) m_velocity[i].setZero();
	
	unsigned c = 0;
	for(j=0; j<= NV; j++) {
		for(i=0; i <= NU; i++) {
			m_pos[c] = Vector3F((float)i * gridSize, iniHeight, (float)j * gridSize);
			m_posLast[c] = m_pos[c];
			m_projectedPos[c] = m_pos[c];
			c++;
		}
	}
/*
2 3
0 1
*/
	const unsigned nl = NU + 1;
	unsigned * id = &m_indices[0];
	for(j=0; j< NV; j++) {
		for(i=0; i < NU; i++) {
			int i0 = j * nl + i;
			int i1 = i0 + 1;
			int i2 = i0 + nl;
			int i3 = i2 + 1;
			if ((j+i)%2) {
				*id++ = i0; *id++ = i2; *id++ = i1;
				*id++ = i1; *id++ = i2; *id++ = i3;
			} else {
				*id++ = i0; *id++ = i2; *id++ = i3;
				*id++ = i0; *id++ = i3; *id++ = i1;
			}
		}
	}
	
	m_numDistanceConstraint = 0;
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU; i++) {
			m_numDistanceConstraint++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i <= NU; i++) {
			m_numDistanceConstraint++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i < NU; i++) {
			m_numDistanceConstraint += 1;
		}
	}
	
	m_numBendingConstraint = 0;
/*
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU - 1; i++) {
			m_numBendingConstraint++;
		}
	}
	
	for(j=0; j< NV - 1; j++) {
		for(i=0; i <= NU; i++) {
			m_numBendingConstraint++;
		}
	}
*/	
	qDebug()<<"num distance constraint "<<m_numDistanceConstraint;
	qDebug()<<"num bending constraint "<<m_numBendingConstraint;
	
	m_distanceConstraint = new pbd::DistanceConstraint[m_numDistanceConstraint];
	
	pbd::DistanceConstraint *d = &m_distanceConstraint[0];
	
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU; i++) {
			setDistanceConstraint(d, nl * j + i, nl * j + i + 1, kStretch);
			d++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i <= NU; i++) {
			setDistanceConstraint(d, nl * j + i, nl * (j + 1) + i, kStretch);
			d++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i < NU; i++) {
			if ((j+i)%2) setDistanceConstraint(d, nl * j + i, nl * (j + 1) + i + 1, kStretch);
			else setDistanceConstraint(d, nl * j + i + 1, nl * (j + 1) + i, kStretch);
			d++;
			//setDistanceConstraint(d, nl * j + i + 1, nl * (j + 1) + i, kStretch);
			//d++;
		}
	}
/*	
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU - 1; i++) {
			setSpring(spr, nl * j + i, nl * j + i + 2, KsBend, KdBend, BEND_SPRING);
			spr++;
		}
	}
	
	for(j=0; j< NV - 1; j++) {
		for(i=0; i <= NU; i++) {
			setSpring(spr, nl * j + i, nl * (j + 2) + i, KsBend, KdBend, BEND_SPRING);
			spr++;
		}
	}
*/	
	qDebug()<<"total mass "<<mass * NP;
	
	m_program = new BoxProgram;
}

SolverThread::~SolverThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();
    wait();
    
}

void SolverThread::initProgram()
{
    m_program->createCvs(NP);
    m_program->createIndices(NI, m_indices);
    m_program->createAabbs(NTri);
}

void SolverThread::simulate()
{
    QMutexLocker locker(&mutex);

    if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        //qDebug()<<"wait";
        condition.wakeOne();
    }
}

void SolverThread::run()
{
   forever {
        // qDebug()<<"run";
	
	for(int i=0; i< 24; i++) {
	    if(restart) {
	        // qDebug()<<"restart ";
            // break;
	    }
	        
	    if (abort) {
            // destroySolverData();
            qDebug()<<"abort";
            return;
        }
        
	    stepPhysics(timeStep);
	}
 
		//if (!restart) {
		    // qDebug()<<"end";
            
		    emit doneStep();
		//}

		mutex.lock();
		
        if (!restart)
            condition.wait(&mutex);
			
        restart = false;
        mutex.unlock();
   }
}

void SolverThread::computeForces() 
{
	unsigned i;
	
	for(i=0;i< NP;i++) {
		m_force[i].setZero();
		if(m_invMass[i] > 0.f) 
		    m_force[i] += gravity / m_invMass[i];
	}
}

void SolverThread::integrateExplicitWithDamping(float dt)
{
	unsigned i = 0;
	
	Vector3F Xcm(0.f, 0.f, 0.f);
	Vector3F Vcm(0.f, 0.f, 0.f);
	float sumM = 0;
	for(i=0;i< NP; i++) {

		m_velocity[i] *= global_dampening; //global velocity dampening !!!		
		m_velocity[i] += (m_force[i] * dt) * m_invMass[i];
		
		// qDebug()<<" acc "<<m_force[i].length() * m_invMass[i];
		
		//calculate the center of mass's position 
		//and velocity for damping calc
		Xcm += m_pos[i];
		Vcm += m_velocity[i];
		sumM += 1.f;
	}
	Xcm /= sumM; 
	Vcm /= sumM;
	
	// qDebug()<<Vcm.str().c_str();	
	
	Matrix33F I; I.setIdentity();
	Vector3F L; L.setZero();
	Vector3F w; w.setZero();//angular velocity
	Matrix33F tmp, tt;
	
	for(i=0;i< NP; i++) { 
		m_Ri[i] = m_pos[i] - Xcm;	
		
		L += m_Ri[i].cross(m_velocity[i]) * mass; 

		*tmp.m(0, 0) = 0.f;			*tmp.m(0, 1) = -m_Ri[i].z;	*tmp.m(0, 2) = m_Ri[i].y; 
		*tmp.m(1, 0) = m_Ri[i].z;	*tmp.m(1, 1) = 0.f;			*tmp.m(1, 2) = -m_Ri[i].x; 
		*tmp.m(2, 0) = -m_Ri[i].y;	*tmp.m(2, 1) = m_Ri[i].x;	*tmp.m(2, 2) = 0.f; 
		
		tt = tmp; tt.transpose();
		tmp.multiply(tt);
		I = I + tmp;
	} 
	
	I.inverse();
	w = I * L;
	
	// qDebug()<<w.str().c_str();
	
	//apply center of mass damping
	for(i=0;i< NP;i++) {
		Vector3F delVi = Vcm + w.cross(m_Ri[i]) - m_velocity[i];		
		m_velocity[i] += delVi * kDamp;
	}

	//calculate predicted position
	for(i=0;i< NP;i++) {
		if(m_invMass[i] <= 0.f) { 
			m_projectedPos[i] = m_pos[i]; //fixed points
		} else {
			m_projectedPos[i] = m_pos[i] + (m_velocity[i] * dt);				 
		}
	} 
}
	
void SolverThread::updateConstraints(float dt)
{
	unsigned i, j;
	for(i = 0; i < solver_iterations; i++) {
		for(j = 0; j < m_numDistanceConstraint; j++) 
			updateDistanceConstraint(j);
		groundCollision();
	}
}

void SolverThread::updateDistanceConstraint(unsigned i)
{
	pbd::DistanceConstraint &c = m_distanceConstraint[i];
	
	Vector3F dir = m_projectedPos[c.p1] - m_projectedPos[c.p2];
	
	float len = dir.length(); 
	if(len <= EPSILON) 
		return;
	
	float w1 = m_invMass[c.p1];
	float w2 = m_invMass[c.p2];
	float invMass = w1 + w2; 
	if(invMass <= EPSILON) 
		return;
 
	Vector3F dP = (dir / len) * (len - c.rest_length) * c.k_prime;
	
	// qDebug()<<" "<<invMass<<" "<<dP.length()<<" "<<len - c.rest_length<<" "<<w1/invMass<<" "<<w2/invMass;

	m_projectedPos[c.p1] -= dP*w1/invMass;

	m_projectedPos[c.p2] += dP*w2/invMass;
}

void SolverThread::groundCollision()
{
	for(unsigned i=0;i< NP;i++) {	
		if(m_projectedPos[i].y<0) //collision with ground
			m_projectedPos[i].y=0;
	}
}

void SolverThread::integrate(float deltaTime) 
{	
	float inv_dt = 1.0f/deltaTime;
	unsigned i; 

	for(i=0;i< NP;i++) {	
		m_velocity[i] = (m_projectedPos[i] - m_pos[i])*inv_dt;		
		m_pos[i] = m_projectedPos[i];		 
	}
}

/*
{		
	for(i=0;i< NP;i++) {
		m_force[i].setZero();
		Vector3F V = getVerletVelocity(m_pos[i], m_posLast[i], dt);
		//add gravity force
		if(i!=0 && i!=NU)	 
			m_force[i] += gravity*mass;
		//add force due to damping of velocity
		m_force[i] += V * DEFAULT_DAMPING;
	}
	
	// float tf = 0.f;
	for(i=0;i< m_numSpring; i++) {
		pbd::Spring & spring = m_spring[i];
		Vector3F p1 = m_pos[spring.p1];
		Vector3F p1Last = m_posLast[spring.p1];
		Vector3F p2 = m_pos[spring.p2];
		Vector3F p2Last = m_posLast[spring.p2];

		Vector3F v1 = getVerletVelocity(p1, p1Last, dt);
		Vector3F v2 = getVerletVelocity(p2, p2Last, dt);

		Vector3F deltaP = p1-p2;
		Vector3F deltaV = v1-v2;
		float dist = deltaP.length();
		if(dist < EPSILON) continue;

		float leftTerm = -spring.Ks * (dist - spring.rest_length);
		// tf += leftTerm;
		float rightTerm = spring.Kd * (deltaV.dot(deltaP)/dist);
		Vector3F springForce = deltaP.normal() * (leftTerm + rightTerm);

		if(spring.p1 != 0 && spring.p1 != NU)
			m_force[spring.p1] += springForce;
		if(spring.p2 != 0 && spring.p2 != NU )
			m_force[spring.p2] -= springForce;
	}
	// qDebug()<<" "<<tf;
}
*/
void SolverThread::integrateVerlet(float deltaTime) 
{
	float deltaTime2Mass = (deltaTime*deltaTime) / mass;
	unsigned i=0;


	for(i=0;i< NP;i++) {
		Vector3F buffer = m_pos[i];

		m_pos[i] = m_pos[i] + (m_pos[i] - m_posLast[i]) + m_force[i] * deltaTime2Mass;

		m_posLast[i] = buffer;

		if(m_pos[i].y <0) {
			m_pos[i].y = 0;
		}
	}
}

Vector3F SolverThread::getVerletVelocity(Vector3F x_i, Vector3F xi_last, float dt ) 
{
	return  (x_i - xi_last) / dt;
}

void SolverThread::stepPhysics(float dt)
{
    computeForces();
	integrateExplicitWithDamping(dt);
	updateConstraints(dt);
	integrate(dt);
	
	// m_program->run(m_pos, NTri, NP);
}

void SolverThread::setSpring(pbd::Spring * dest, unsigned a, unsigned b, float ks, float kd, int type) 
{
	dest->p1=a;
	dest->p2=b;
	dest->Ks=ks;
	dest->Kd=kd;
	dest->type = type;
	Vector3F deltaP = m_pos[a]-m_pos[b];
	dest->rest_length = deltaP.length();
}

void SolverThread::setDistanceConstraint(pbd::DistanceConstraint * dest, unsigned a, unsigned b, float k)
{
	dest->p1 = a;
	dest->p2 = b;
	dest->k_prime = 1.0f - pow((1.f - k), 1.f / (float)solver_iterations);
	if(dest->k_prime > 1.f) dest->k_prime = 1.f;

	Vector3F deltaP = m_pos[a]-m_pos[b];
	dest->rest_length = deltaP.length();
}

unsigned SolverThread::numIndices() const { return NI; }
Vector3F * SolverThread::pos() { return m_pos; }
unsigned * SolverThread::indices() { return m_indices; }
