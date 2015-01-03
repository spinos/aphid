#include <QtCore>
#include "SolverThread.h"
#include <BoxProgram.h>
#define NU 50
#define NV 50
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
float timeStep = 1.f / 60.f;
float divG = 9.81f / 24.f;
Vector3F gravity(0.0f,-divG,0.0f);
float mass = 1.f;
float iniHeight = 24.f;
float gridSize = 24.f / (float)NU;

SolverThread::SolverThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;
    
    m_pos = new Vector3F[NP];
	m_posLast = new Vector3F[NP];
	m_force = new Vector3F[NP];
	m_indices = new unsigned[NI];
	
	unsigned i, j;
	unsigned c = 0;
	for(j=0; j<= NV; j++) {
		for(i=0; i <= NU; i++) {
			m_pos[c] = Vector3F((float)i * gridSize, iniHeight, (float)j * gridSize);
			m_posLast[c] = m_pos[c];
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
	
	m_numSpring = 0;
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU; i++) {
			m_numSpring++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i <= NU; i++) {
			m_numSpring++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i < NU; i++) {
			m_numSpring += 2;
		}
	}
	
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU - 1; i++) {
			m_numSpring++;
		}
	}
	
	for(j=0; j< NV - 1; j++) {
		for(i=0; i <= NU; i++) {
			m_numSpring++;
		}
	}
	
	qDebug()<<"num spring "<<m_numSpring;
	m_spring = new pbd::Spring[m_numSpring];
	
	pbd::Spring *spr = &m_spring[0];
	
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU; i++) {
			setSpring(spr, nl * j + i, nl * j + i + 1, KsStruct, KdStruct, STRUCTURAL_SPRING);
			spr++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i <= NU; i++) {
			setSpring(spr, nl * j + i, nl * (j + 1) + i, KsStruct, KdStruct, STRUCTURAL_SPRING);
			spr++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i < NU; i++) {
			setSpring(spr, nl * j + i, nl * (j + 1) + i + 1, KsShear, KdShear, SHEAR_SPRING);
			spr++;
			setSpring(spr, nl * j + i + 1, nl * (j + 1) + i, KsShear, KdShear, SHEAR_SPRING);
			spr++;
		}
	}
	
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
	
	for(int i=0; i< 30; i++) {
	    if(restart) {
	        // qDebug()<<"restart ";
            break;
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

void SolverThread::computeForces(float dt) 
{
	unsigned i=0;

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
	computeForces(dt);
	integrateVerlet(dt);
	m_program->run(m_pos, NTri, NP);
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
unsigned SolverThread::numIndices() const { return NI; }
Vector3F * SolverThread::pos() { return m_pos; }
unsigned * SolverThread::indices() { return m_indices; }
