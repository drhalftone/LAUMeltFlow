#pragma once

#include <QWidget>
#include <QVector>
#include <QPointF>

class ChainSimulator;

/// Widget that renders the bead chain in real time using QPainter.
///
/// Coordinate system: simulation uses physics coords (x right, y up).
/// The widget maps these to screen coords (y flipped, centered and scaled).
///
class ChainWidget : public QWidget {
    Q_OBJECT

public:
    explicit ChainWidget(ChainSimulator *sim, QWidget *parent = nullptr);

    /// Map from simulation (physics) coordinates to widget pixels.
    QPointF toScreen(float sx, float sy) const;

public slots:
    void onSimStepped();

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    ChainSimulator *m_sim;

    // View parameters (auto-fit to chain extent)
    double m_scale = 150.0;    // pixels per meter
    double m_centerX = 0.0;    // sim-space center x
    double m_centerY = -1.0;   // sim-space center y
};
