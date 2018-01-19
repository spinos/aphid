/*
 *  lbm collision
 */
#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <qt/Base3DView.h>

namespace aphid {

template<typename T>
class DenseVector;

namespace lbm {

class VolumeResponse;
class LatticeBlock;
}

}

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:

    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	
protected:
	virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(aphid::Vector3F & origin, aphid::Vector3F & ray, aphid::Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(aphid::Vector3F & stir);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    virtual void resetPerspViewTransform();
	virtual void resetOrthoViewTransform();
	
private:
	void drawBlock(aphid::lbm::LatticeBlock* blk);
	void simulationStep(bool toMoveParticles);
	
private:
    aphid::lbm::VolumeResponse* m_latman;
	aphid::DenseVector<float>* m_nodeCenter;
	aphid::DenseVector<float>* m_nodeU;
	aphid::DenseVector<float>* m_nodeRho;
	aphid::DenseVector<float>* m_particleX;
	aphid::DenseVector<float>* m_particleU;
	aphid::DenseVector<float>* m_particleUhat;
	
};

#endif
