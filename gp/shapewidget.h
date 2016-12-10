#ifndef INST_GLWIDGET_H
#define INST_GLWIDGET_H
#include <QGLWidget>
#include <Base3DView.h>
#include <AllMath.h>
#include <boost/scoped_array.hpp>

namespace aphid {
class SuperQuadricGlyph;
class GlslLegacyInstancer;

template<typename T>
class DenseMatrix;

namespace gpr {
template<typename T>
class RbfKernel;

template<typename T>
class RbfKernel;

template<typename T, typename T1>
class Covariance;

}

}

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	
	const aphid::DenseMatrix<float > & K() const;

protected:
    virtual void clientInit();
    virtual void clientDraw();
	virtual void clientMouseInput(QMouseEvent *event);

public slots:

signals:
    
private:
	void predict(const float & x0, const float & x1);

typedef aphid::SuperQuadricGlyph * GlyphPtrType;

    boost::scoped_array<GlyphPtrType > m_glyphs;
    boost::scoped_array<aphid::Float4> m_particles;
    aphid::GlslLegacyInstancer * m_instancer;
    int m_numParticles;

	aphid::DenseMatrix<float> * m_xMeasure;
	aphid::DenseMatrix<float> * m_yMeasure;
	aphid::gpr::Covariance<float, aphid::gpr::RbfKernel<float> > * m_covTrain;
	aphid::DenseMatrix<float> * m_xPredict;
	aphid::DenseMatrix<float> * m_yPredict;
	aphid::gpr::RbfKernel<float> * m_rbf;
};

#endif
