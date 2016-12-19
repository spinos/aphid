#include <QtGui>
#include "drpcawidget.h"
#include "drpcawindow.h"

Window::Window()
{
	 qDebug()<<"dimensionality reduction PCA ";
	 QDateTime local(QDateTime::currentDateTime());
	 qDebug() << "local time is:" << local;
	 srand (local.toTime_t() );
    glWidget = new GLWidget(this);

	setCentralWidget(glWidget);
    setWindowTitle(tr("Dimensionality Reduction PCA"));
}

Window::~Window()
{ qDebug()<<"exit dimensionality reduction pca window"; }

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

