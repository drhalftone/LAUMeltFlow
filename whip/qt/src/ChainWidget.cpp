#include "ChainWidget.h"
#include "ChainSimulator.h"
#include "Bead.h"

#include <QPainter>
#include <QPen>
#include <QBrush>
#include <cmath>

ChainWidget::ChainWidget(ChainSimulator *sim, QWidget *parent)
    : QWidget(parent)
    , m_sim(sim)
{
    setMinimumSize(600, 600);
    setAutoFillBackground(true);

    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(30, 30, 30));
    setPalette(pal);
}

QPointF ChainWidget::toScreen(float sx, float sy) const
{
    // Physics → screen: center in widget, flip y, apply scale
    double px = width() / 2.0 + (sx - m_centerX) * m_scale;
    double py = height() / 2.0 - (sy - m_centerY) * m_scale;
    return QPointF(px, py);
}

void ChainWidget::onSimStepped()
{
    update(); // schedule repaint
}

void ChainWidget::paintEvent(QPaintEvent * /*event*/)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int n = m_sim->beadCount();
    if (n == 0)
        return;

    // ── Draw rods (edges) ──────────────────────────────────────────
    {
        QPen rodPen(QColor(100, 180, 255, 200), 2.0);
        p.setPen(rodPen);

        for (const Rod &rod : m_sim->rods()) {
            const Bead *a = m_sim->bead(rod.beadA);
            const Bead *b = m_sim->bead(rod.beadB);
            QPointF pa = toScreen(a->pos().x(), a->pos().y());
            QPointF pb = toScreen(b->pos().x(), b->pos().y());
            p.drawLine(pa, pb);
        }
    }

    // ── Draw beads (nodes) ─────────────────────────────────────────
    {
        double radius = 5.0;
        p.setPen(Qt::NoPen);

        for (int i = 0; i < n; ++i) {
            const Bead *bead = m_sim->bead(i);
            QPointF sp = toScreen(bead->pos().x(), bead->pos().y());

            if (bead->isFixed()) {
                // Anchor: red square
                p.setBrush(QColor(255, 80, 80));
                p.drawRect(QRectF(sp.x() - radius, sp.y() - radius,
                                  2 * radius, 2 * radius));
            } else {
                // Free bead: white circle, slight gradient toward tip
                int t = 255 - (i * 120 / n); // brighter near anchor
                p.setBrush(QColor(t, t, 255));
                p.drawEllipse(sp, radius, radius);
            }
        }
    }

    // ── Draw ceiling line ──────────────────────────────────────────
    {
        const Bead *anchor = m_sim->bead(0);
        QPointF anchorScreen = toScreen(anchor->pos().x(), anchor->pos().y());
        QPen ceilPen(QColor(120, 120, 120), 1.0, Qt::DashLine);
        p.setPen(ceilPen);
        p.drawLine(QPointF(0, anchorScreen.y()),
                   QPointF(width(), anchorScreen.y()));
    }

    // ── Info text ──────────────────────────────────────────────────
    {
        const Bead *tip = m_sim->bead(n - 1);
        double tipSpeed = std::sqrt(tip->vel().x() * tip->vel().x()
                                  + tip->vel().y() * tip->vel().y());

        p.setPen(QColor(200, 200, 200));
        p.setFont(QFont("Consolas", 10));
        p.drawText(10, 20, QString("t = %1 s  |  tip: %2 m/s")
                   .arg(m_simTime, 0, 'f', 3)
                   .arg(tipSpeed, 0, 'f', 1));
    }
}
