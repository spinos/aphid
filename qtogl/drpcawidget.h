#ifndef DR_PCA_WIDGET_H
#define DR_PCA_WIDGET_H

#include <Base3DView.h>

namespace aphid {

template<typename T>
class DenseMatrix;

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

public slots:

signals:
    
private:
#define MAX_N_DATA 100
	aphid::DenseMatrix<float > * m_data[MAX_N_DATA];
	int m_N;
	int m_D;
};

#endif
