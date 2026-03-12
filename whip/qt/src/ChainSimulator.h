#pragma once

#include <QObject>
#include <QVector>
#include <QVector2D>

class Bead;

/// Describes one rod element (edge) between two beads.
struct Rod {
    int beadA;
    int beadB;
    double restLength;
};

/// Coordinator that owns all beads and rods, wires up signals/slots,
/// and drives the two-phase timestep:
///   Phase 1: broadcast neighbor states → beads accumulate forces
///   Phase 2: all beads integrate simultaneously
///
class ChainSimulator : public QObject {
    Q_OBJECT

public:
    explicit ChainSimulator(QObject *parent = nullptr);

    /// Build a horizontal chain of n beads, fixed at the left end.
    void buildChain(int nBeads, double totalLength, double totalMass,
                    double stiffness, double damping, double drag = 0.0,
                    QVector2D anchorPos = QVector2D(0, 0));

    /// Advance the simulation by one timestep.
    void step(double dt, double gravity);

    // Accessors for the renderer
    int beadCount() const { return m_beads.size(); }
    const Bead *bead(int i) const { return m_beads[i]; }
    const QVector<Rod> &rods() const { return m_rods; }

signals:
    /// Emitted after each step so the widget can repaint.
    void stepped();

private:
    QVector<Bead *> m_beads;
    QVector<Rod> m_rods;
};
