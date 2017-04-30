#ifndef GP_DF_X_DIALOG_H
#define GP_DF_X_DIALOG_H
#include <QDialog>

class GpdfxWidget;

class GpdfxDialog : public QDialog
{
    Q_OBJECT

public:
    GpdfxDialog(QWidget *parent = 0);

protected:
    
public slots:
   void recvXValue(QPointF x);

signals:
	void xValueChanged(QPointF x);
	
private:

private:
	GpdfxWidget * m_wig;
	
};
#endif