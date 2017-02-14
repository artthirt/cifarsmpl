#ifndef DRAWCNVWEIGHT_H
#define DRAWCNVWEIGHT_H

#include <QWidget>

#include "custom_types.h"

namespace Ui {
class DrawCnvWeight;
}

class DrawCnvWeight : public QWidget
{
	Q_OBJECT

public:
	explicit DrawCnvWeight(QWidget *parent = 0);
	~DrawCnvWeight();

	void set_weightR(const QVector<QVector<ct::Matf> > &W);
	void set_weightG(const QVector<  QVector < ct::Matf > > &W);
	void set_weightB(const QVector<  QVector < ct::Matf > > &W);

private:
	Ui::DrawCnvWeight *ui;

	// QWidget interface
protected:
	virtual void paintEvent(QPaintEvent *event);

private:
	QVector< QVector < ct::Matf > > m_W_R, m_prevW_R;
	QVector< QVector < ct::Matf > > m_W_G, m_prevW_G;
	QVector< QVector < ct::Matf > > m_W_B, m_prevW_B;

	QSize draw_weight(QPainter& painter, int offset);
};

#endif // DRAWCNVWEIGHT_H
