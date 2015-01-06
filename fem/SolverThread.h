#ifndef SOLVERTHREAD_H
#define SOLVERTHREAD_H
#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>

#include <AllMath.h>
struct Tetrahedron {
	unsigned indices[4];			//indices
	float volume;			//volume 
	float plastic[6];		//plasticity values
	Matrix33F Re;			//Rotational warp of tetrahedron.
	Matrix33F Ke[4][4];		//Stiffness element matrix
	Vector3F e1, e2, e3;	//edges
	Vector3F B[4];			//Jacobian of shapefunctions; B=SN =[d/dx  0     0 ][wn 0  0]
							//                                  [0    d/dy   0 ][0 wn  0]
							//									[0     0   d/dz][0  0 wn]
							//									[d/dy d/dx   0 ]
							//									[d/dz  0   d/dx]
							//									[0    d/dz d/dy]
};

#include <map>
typedef std::map<int, Matrix33F> MatrixMap;

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE
class BoxProgram;
class SolverThread : public QThread
{
    Q_OBJECT

public:
    SolverThread(QObject *parent = 0);
    ~SolverThread();
    
    unsigned numTetrahedrons() const;
    Tetrahedron * tetrahedron();
    Vector3F * X();

signals:
    void doneStep();

protected:
    void run();

private:
    QMutex mutex;
    QWaitCondition condition;
    
    Vector3F * m_X;
	Vector3F * m_Xi;
	bool * m_IsFixed;
	Vector3F * m_F;
	Vector3F * m_F0;
	Vector3F * m_b;
	Vector3F * m_V;
	Vector3F * m_residual;
	Vector3F * m_update;
	Vector3F * m_prev;
	MatrixMap * m_K_row;
	MatrixMap * m_A_row;
	Tetrahedron * m_tetrahedron;
	
	float * m_mass;
	unsigned m_totalPoints, m_totalTetrahedrons;
	
    bool restart;
    bool abort;
    
    void generateBlocks(unsigned xdim, unsigned ydim, unsigned zdim, float width, float height, float depth);
	void addTetrahedron(Tetrahedron *t, unsigned i0, unsigned i1, unsigned i2, unsigned i3);
    void calculateK();
    void clearStiffnessAssembly();
    void recalcMassMatrix();
    void initializePlastic();
    
    void computeForces();
    void updateOrientation();
    void resetOrientation();
    
    void stepPhysics(float dt);
    
    static float getTetraVolume(Vector3F e1, Vector3F e2, Vector3F e3);
	
public slots:
    void simulate();
	
};

#endif        //  #ifndef SOLVERTHREAD_H

