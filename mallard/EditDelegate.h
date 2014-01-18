#pragma once

#include <QStyledItemDelegate>
#include <QModelIndex>
#include <QObject>
#include <QSize>
#include <QSpinBox>
#include <QIntValidator>
#include <QDoubleValidator>

class EditDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    EditDelegate(QObject *parent = 0);

	void paint(QPainter *painter, const QStyleOptionViewItem &option,
               const QModelIndex &index) const;
			   
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                          const QModelIndex &index) const;

    void setEditorData(QWidget *editor, const QModelIndex &index) const;
    void setModelData(QWidget *editor, QAbstractItemModel *model,
                      const QModelIndex &index) const;

private slots:

private:
	QWidget * createIntEditor(QWidget *parent) const;
	QWidget * createDoubleEditor(QWidget *parent) const;
	QWidget * createBoolEditor(QWidget *parent) const;
	QWidget * createColorEditor(QWidget *parent, QColor col) const;
	
	void setIntEditorValue(QWidget *editor, QVariant & value) const;
	void setDoubleEditorValue(QWidget *editor, QVariant & value) const;
	void setBoolEditorValue(QWidget *editor, QVariant & value) const;
	void setColorEditorValue(QWidget *editor, QVariant & value) const;
	
	QVariant getIntEditorValue(QWidget *editor) const;
	QVariant getDoubleEditorValue(QWidget *editor) const;
	QVariant getBoolEditorValue(QWidget *editor) const;
	QVariant getColorEditorValue(QWidget *editor) const;
	
	QString translateBoolToStr(const QVariant & src) const;
	QVariant translateStrToBool(const QString & src) const;
	QIntValidator m_intValidate;
	QDoubleValidator m_dlbValidate;
};

