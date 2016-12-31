#ifndef NACA_PARAM_DIALOG_H
#define NACA_PARAM_DIALOG_H
#include <QDialog>
#include <math/ATypes.h>

class ParamWidget;

class ParamDialog : public QDialog
{
    Q_OBJECT

public:
    ParamDialog(QWidget *parent = 0);

protected:
    
public slots:
   void recvCamber(double x);
   void recvPosition(double x);
   void recvThickness(double x);
   
signals:
	void paramChanged(aphid::Float3 x);
	
private:

private:
	aphid::Float3 m_cpt;
	ParamWidget * m_wig;
	
};
#endif