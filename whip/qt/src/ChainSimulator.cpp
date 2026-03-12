#include "ChainSimulator.h"
#include "Bead.h"

ChainSimulator::ChainSimulator(QObject *parent)
    : QObject(parent)
{
}

void ChainSimulator::buildChain(int nBeads, double totalLength,
                                double totalMass, double stiffness,
                                double damping, double drag,
                                QVector2D anchorPos)
{
    // Clean up any previous chain
    qDeleteAll(m_beads);
    m_beads.clear();
    m_rods.clear();

    double L0 = totalLength / (nBeads - 1);
    double massPerBead = totalMass / nBeads;

    // Create beads
    for (int i = 0; i < nBeads; ++i) {
        auto *bead = new Bead(i, massPerBead, stiffness, damping, drag, this);
        bead->setPos(anchorPos + QVector2D(static_cast<float>(i * L0), 0));
        bead->setVel(QVector2D(0, 0));
        if (i == 0)
            bead->setFixed(true);
        m_beads.append(bead);
    }

    // Create rod elements and wire up signal/slot connections
    for (int i = 0; i < nBeads - 1; ++i) {
        Rod rod;
        rod.beadA = i;
        rod.beadB = i + 1;
        rod.restLength = L0;
        m_rods.append(rod);

        // Bidirectional connections — each bead tells the other its state.
        // We don't connect stateChanged directly because we need the
        // coordinator to drive the phases. Instead we call
        // onNeighborState explicitly in step().
    }
}

void ChainSimulator::step(double dt, double gravity)
{
    // ── Phase 1: Force accumulation (message passing) ──────────────

    // Reset all force accumulators
    for (Bead *b : m_beads)
        b->resetForces();

    // Apply gravity to every bead
    for (Bead *b : m_beads)
        b->applyGravity(gravity);

    // Send neighbor messages along each rod (bidirectional)
    for (const Rod &rod : m_rods) {
        Bead *a = m_beads[rod.beadA];
        Bead *b = m_beads[rod.beadB];

        // a tells b its state
        b->onNeighborState(a->id(), a->pos(), a->vel(), rod.restLength);
        // b tells a its state
        a->onNeighborState(b->id(), b->pos(), b->vel(), rod.restLength);
    }

    // ── Phase 2: Integration (node update) ─────────────────────────

    for (Bead *b : m_beads)
        b->integrate(dt);

    emit stepped();
}
