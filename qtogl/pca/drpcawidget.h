#ifndef DR_PCA_WIDGET_H
#define DR_PCA_WIDGET_H

#include <qt/Base3DView.h>

namespace aphid {

template<typename T>
class DenseMatrix;

template<typename T, int N>
class PCAFeature;

template<typename T, typename Tf>
class PCASimilarity;

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
	void drawFeatures();
    
private:
	aphid::PCASimilarity<float, aphid::PCAFeature<float, 3> > * m_features;
	
#define MAX_N_DATA 100
	aphid::DenseMatrix<float > * m_data[MAX_N_DATA];
	int m_N;
	int m_D;
};

#endif
