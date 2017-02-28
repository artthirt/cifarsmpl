#ifndef IMUTILS_H
#define IMUTILS_H

#include "custom_types.h"

#include <QVector>
#include <QPainter>

void search_minmax(const QVector< ct::Matf > &Ws, float& vmin, float &vmax);

void normalizeMat(const ct::Matf& iW, ct::Matf& oW, float _min, float _max);

void draw_W(QPainter& painter, const ct::Matf& W_R,
			const ct::Matf& W_G,
			const ct::Matf& W_B,
			int _x, int _y, int w, float _max, float _min);

void draw_W(QPainter& painter, const ct::Matf& W, int _x, int _y, int w, float _max, float _min);



#endif // IMUTILS_H
