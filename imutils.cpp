#include "imutils.h"

#include "matops.h"

void search_minmax(const QVector< ct::Matf > &Ws, float& vmin, float &vmax)
{
	vmax = -99999999.f, vmin = 999999999.f;

	for(int j = 0; j < Ws.size(); ++j){
		float m1 = Ws[j].max();
		float m2 = Ws[j].min();

		vmax = std::max(m1, vmax);
		vmin = std::min(m2, vmin);
	}
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
