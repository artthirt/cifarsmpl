#include "drawcnvweight.h"
#include "ui_drawcnvweight.h"

#include <QPaintEvent>
#include <QPainter>

#include "matops.h"

const int wd_blk = 10;

DrawCnvWeight::DrawCnvWeight(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::DrawCnvWeight)
{
	ui->setupUi(this);
	setMouseTracking(true);
	m_update = false;

	m_offset = 0;

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(200);
}

DrawCnvWeight::~DrawCnvWeight()
{
	delete ui;
}

void copy_weights(const  QVector< QVector<ct::Matf> > &W,
				   QVector< QVector<ct::Matf> > &Wout)
{
	Wout.clear();
	Wout.resize(W.size());
	for(size_t i = 0; i < W.size(); i++){
		Wout[i].resize(W[i].size());
		for(size_t j = 0; j < W[i].size(); j++){
			W[i][j].copyTo(Wout[i][j]);
		}
	}
}

void DrawCnvWeight::set_weightR(const QVector<QVector<ct::Matf> > &W)
{
	if(W.empty())
		return;

	copy_weights(W, m_W_R);

	if(m_prevW_R.empty())
		copy_weights(m_W_R, m_prevW_R);
}

void DrawCnvWeight::set_weightG(const QVector< QVector<ct::Matf> > &W)
{
	if(W.empty())
		return;

	copy_weights(W, m_W_G);

	if(m_prevW_G.empty())
		copy_weights(m_W_G, m_prevW_G);

	update();
}

void DrawCnvWeight::set_weightB(const QVector< QVector<ct::Matf> > &W)
{
	if(W.empty())
		return;

	copy_weights(W, m_W_B);

	if(m_prevW_B.empty())
		copy_weights(m_W_B, m_prevW_B);

	update();
}

void DrawCnvWeight::onTimeout()
{
	if(m_update){
		m_update = false;
		update();
	}
}

void DrawCnvWeight::set_weightGray(const QVector<QVector<ct::Matf> > &W)
{
	if(W.empty())
		return;

	copy_weights(W, m_W_Gray);

	if(m_prevW_gray.empty())
		copy_weights(m_W_Gray, m_prevW_gray);

	update();
}

void normalizeMat(const ct::Matf& iW, ct::Matf& oW, float _min, float _max)
{
	float m1 = _max;
	float m2 = _min;
	oW = iW - m2;
	oW *= (1./(m1 - m2));
	oW *= 255.;
}

void draw_W(QPainter& painter, const ct::Matf& W_R,
			const ct::Matf& W_G,
			const ct::Matf& W_B,
			int _x, int _y, int w, float _max, float _min)
{
	ct::Matf Wr, Wg, Wb;

	normalizeMat(W_R, Wr, _min, _max);
	normalizeMat(W_G, Wg, _min, _max);
	normalizeMat(W_B, Wb, _min, _max);

//	float *dW1 = W.ptr();
//	float *dW2 = prevW.ptr();

	float *dWR = Wr.ptr();
	float *dWG = Wg.ptr();
	float *dWB = Wb.ptr();

	for(int y = 0; y < Wr.rows; ++y){
		for(int x = 0; x < Wr.cols; ++x){
			QRect rt(QPoint(_x + x * w, _y + y * w), QSize(w, w));
			uchar r = dWR[y * Wr.cols + x];
			uchar g = dWG[y * Wg.cols + x];
			uchar b = dWB[y * Wb.cols + x];
			painter.setBrush(QColor(r, g, b));
			painter.drawRect(rt);
		}
	}
}

void draw_W(QPainter& painter, const ct::Matf& W, int _x, int _y, int w, float _max, float _min)
{
	ct::Matf _W;
	normalizeMat(W, _W, _min, _max);

//	float *dW1 = W.ptr();
//	float *dW2 = prevW.ptr();

	float *dW = _W.ptr();
	for(int y = 0; y < _W.rows; ++y){
		for(int x = 0; x < _W.cols; ++x){
			QRect rt(QPoint(_x + x * w, _y + y * w), QSize(w, w));
			uchar c = dW[y * _W.cols + x];
			painter.setBrush(QColor(c, c, c));
			painter.drawRect(rt);
		}
	}
}

void DrawCnvWeight::paintEvent(QPaintEvent *event)
{
	Q_UNUSED(event);

	QPainter painter(this);

	painter.fillRect(rect(), Qt::black);

	QSize s;
	s = draw_weight(painter, m_offset);
	draw_weightGray(painter, s.height() + 20);
}

void search_minmax(const QVector< ct::Matf > &Ws, float& vmin, float &vmax)
{
	vmax = -99999999.f, vmin = 999999999.f;

	for(size_t j = 0; j < Ws.size(); ++j){
		float m1 = Ws[j].max();
		float m2 = Ws[j].min();

		vmax = std::max(m1, vmax);
		vmin = std::min(m2, vmin);
	}
}

QSize DrawCnvWeight::draw_weight(QPainter &painter, int offset)
{
	if(m_W_R.empty() || m_W_R.size() != m_W_G.size() || m_W_R.size() != m_W_B.size())
		return QSize(0, 0);

	int x = 0, y = offset, w = 0, h = 0;

	float min1, max1, min2, max2, min3, max3, min0, max0;

	for(size_t i = 0; i < m_W_R.size(); ++i){
		const  QVector< ct::Matf > &Ws1 = m_W_R[i];
		const  QVector< ct::Matf > &Ws2 = m_W_G[i];
		const  QVector< ct::Matf > &Ws3 = m_W_B[i];
		x = 0;

		search_minmax(Ws1, min1, max1);
		search_minmax(Ws2, min2, max2);
		search_minmax(Ws3, min3, max3);

		min0 = std::min(min1, std::min(min2, min3));
		max0 = std::max(max1, std::max(max2, max3));

		for(size_t j = 0; j < Ws1.size(); ++j){
			ct::Matf WR = Ws1[j];
			ct::Matf WG = Ws2[j];
			ct::Matf WB = Ws3[j];

			w = wd_blk * WR.cols;
			h = wd_blk * WR.rows;

			if(x >= rect().width() - (w + 2)){
				x = 0;
				y += h + 2;
			}

			painter.setPen(Qt::NoPen);
			draw_W(painter, WR, WG, WB, x, y, wd_blk, max0, min0);
			painter.setPen(Qt::blue);
			painter.setBrush(Qt::NoBrush);
			painter.drawRect(QRect(QPoint(x, y), QSize(w, h)));

			x += w + 2;
		}
		y += h + 2;
	}
	return QSize(x, y - offset);
}

QSize DrawCnvWeight::draw_weightGray(QPainter &painter, int offset)
{
	if(m_W_Gray.empty())
		return QSize(0, 0);

	int x = 0, y = offset, w = 0, h = 0;

	for(size_t i = 0; i < m_W_Gray.size(); ++i){
		const QVector< ct::Matf > &Ws = m_W_Gray[i];
		x = 0;

		float _max = -99999999.f, _min = 999999999.f;
		search_minmax(Ws, _min, _max);

		for(size_t j = 0; j < Ws.size(); ++j){
			ct::Matf W = Ws[j];

			w = wd_blk * W.cols;
			h = wd_blk * W.rows;

			if(x >= rect().width() - (w + 2)){
				x = 0;
				y += h + 2;
			}

			painter.setPen(Qt::NoPen);
			draw_W(painter, W, x, y, wd_blk, _max, _min);
			painter.setPen(Qt::blue);
			painter.setBrush(Qt::NoBrush);
			painter.drawRect(QRect(QPoint(x, y), QSize(w, h)));

			x += w + 2;
		}
		y += h + 2;
	}
	return QSize(x, y - offset);
}

void DrawCnvWeight::mouseMoveEvent(QMouseEvent *event)
{
	if(event->buttons().testFlag(Qt::LeftButton)){
		m_offset += event->pos().y() - m_pt.y();
		m_pt = event->pos();
		if(m_offset < -2000){
			m_offset = -2000;
		}
		if(m_offset > 200){
			m_offset = 200;
		}
		m_update = true;
	}
}


void DrawCnvWeight::mousePressEvent(QMouseEvent *event)
{
	m_pt = event->pos();
	if(event->buttons().testFlag(Qt::RightButton)){
		m_offset = 0;
		m_update = true;
	}
}
