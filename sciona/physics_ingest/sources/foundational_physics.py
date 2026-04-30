"""Manual foundational physics backfill scaffold.

The records in this module are curated seed candidates for canonical physics
laws. They are intentionally side-effect free: builders emit snapshot and
candidate dictionaries only, leaving DB insertion to downstream loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from sciona.physics_ingest.sources._manifest import (
    JSONDict,
    SourceAdapterBundle,
    build_snapshot_row,
)


SOURCE_SYSTEM = "manual"
ADAPTER_NAME = "sciona.physics_ingest.sources.foundational_physics"
ADAPTER_VERSION = "wave1.foundational_physics_backfill.v1"
SOURCE_VERSION = "curated-foundational-physics-v1"
SOURCE_URI = "manual://sciona.physics_ingest/foundational_physics/v1"
LICENSE_EXPRESSION = (
    "Curated factual formula metadata; references identify source works for "
    "human verification before publication."
)
PROVENANCE_SUMMARY = (
    "Deterministic manual seed set for foundational physics laws across "
    "mechanics, thermodynamics, electromagnetism, waves, diffusion/transport, "
    "and scaling laws."
)


@dataclass(frozen=True)
class FoundationalLawSeed:
    """Curated raw Wave 0 candidate for one foundational law."""

    source_id: str
    domain: str
    label: str
    description: str
    raw_formula: str
    variable_dimension_hints: Mapping[str, str]
    mechanism_tags: tuple[str, ...]
    behavioral_archetypes: tuple[str, ...]
    references: tuple[Mapping[str, str], ...]
    assumptions: tuple[str, ...] = ()
    notes: str = ""
    formula_format: str = "plain_text"
    priority_score: float = 0.75
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self, *, source_uri: str = SOURCE_URI) -> JSONDict:
        return {
            "source_system": SOURCE_SYSTEM,
            "source_kind": "curated_foundational_law",
            "source_id": self.source_id,
            "source_uri": f"{source_uri}/{self.source_id}",
            "domain": self.domain,
            "raw_formula": self.raw_formula,
            "raw_formula_format": self.formula_format,
            "variable_dimension_hints": dict(self.variable_dimension_hints),
            "mechanism_tags": list(self.mechanism_tags),
            "behavioral_archetypes": list(self.behavioral_archetypes),
            "assumptions": list(self.assumptions),
            "references": [dict(reference) for reference in self.references],
            "provenance": {
                "curation_method": "manual_seed",
                "curation_scope": SOURCE_VERSION,
                **dict(self.provenance),
            },
        }

    def to_candidate_row(
        self,
        *,
        source_uri: str = SOURCE_URI,
        snapshot_id: str | None = None,
    ) -> JSONDict:
        row: JSONDict = {
            "source_candidate_id": self.source_id,
            "source_entity_uri": f"{source_uri}/{self.source_id}",
            "source_label": self.label,
            "source_description": self.description,
            "raw_formula": self.raw_formula,
            "raw_formula_format": self.formula_format,
            "candidate_status": "raw_imported",
            "parse_confidence": 0.0,
            "priority_score": self.priority_score,
            "mechanism_tags": list(self.mechanism_tags),
            "behavioral_archetypes": list(self.behavioral_archetypes),
            "source_payload": self.to_payload(source_uri=source_uri),
            "notes": self.notes
            or (
                "Curated foundational seed retained as raw formula; symbolic "
                "normalization, dimensional validation, and publication review "
                "are pending."
            ),
        }
        if snapshot_id:
            row["snapshot_id"] = snapshot_id
        return row


FOUNDATIONAL_LAW_SEEDS: tuple[FoundationalLawSeed, ...] = (
    FoundationalLawSeed(
        source_id="manual:mechanics:newton_second_law",
        domain="mechanics",
        label="Newton's second law",
        description="Net force equals mass times acceleration for a point mass.",
        raw_formula="F = m a",
        variable_dimension_hints={"F": "M1L1T-2", "m": "M1", "a": "L1T-2"},
        mechanism_tags=("force_balance", "inertial_response", "classical_mechanics"),
        behavioral_archetypes=("linear_proportionality", "second_order_dynamics"),
        assumptions=("inertial frame", "constant mass", "nonrelativistic speeds"),
        references=(
            {
                "title": "Philosophiae Naturalis Principia Mathematica",
                "author": "Isaac Newton",
                "year": "1687",
                "provenance_note": "Classical source for Newtonian mechanics.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:mechanics:hooke_law",
        domain="mechanics",
        label="Hooke's law",
        description="Restoring force is proportional to displacement.",
        raw_formula="F = -k x",
        variable_dimension_hints={"F": "M1L1T-2", "k": "M1T-2", "x": "L1"},
        mechanism_tags=("elastic_restoring_force", "linear_response"),
        behavioral_archetypes=("linear_proportionality", "stable_equilibrium"),
        assumptions=("small deformation", "linear elastic regime"),
        references=(
            {
                "title": "Lectures on Physics, Vol. I",
                "author": "Richard P. Feynman; Robert B. Leighton; Matthew Sands",
                "year": "1963",
                "provenance_note": "Textbook reference for linear springs.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:mechanics:universal_gravitation",
        domain="mechanics",
        label="Newtonian universal gravitation",
        description="Point masses attract with inverse-square gravitational force.",
        raw_formula="F = G m1 m2 / r^2",
        variable_dimension_hints={
            "F": "M1L1T-2",
            "G": "M-1L3T-2",
            "m1": "M1",
            "m2": "M1",
            "r": "L1",
        },
        mechanism_tags=("central_force", "inverse_square_law", "gravity"),
        behavioral_archetypes=("inverse_square_decay", "long_range_interaction"),
        assumptions=("point masses or spherical symmetry", "Newtonian regime"),
        references=(
            {
                "title": "Philosophiae Naturalis Principia Mathematica",
                "author": "Isaac Newton",
                "year": "1687",
                "provenance_note": "Classical source for universal gravitation.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:thermodynamics:first_law",
        domain="thermodynamics",
        label="First law of thermodynamics",
        description="Change in internal energy equals heat added minus work done by the system.",
        raw_formula="dU = delta Q - delta W",
        variable_dimension_hints={"U": "M1L2T-2", "Q": "M1L2T-2", "W": "M1L2T-2"},
        mechanism_tags=("energy_conservation", "heat_work_exchange"),
        behavioral_archetypes=("conservation_law", "balance_equation"),
        assumptions=("closed system sign convention stated in payload",),
        references=(
            {
                "title": "Thermodynamics and an Introduction to Thermostatistics",
                "author": "Herbert B. Callen",
                "year": "1985",
                "provenance_note": "Standard thermodynamics textbook reference.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:thermodynamics:ideal_gas_law",
        domain="thermodynamics",
        label="Ideal gas law",
        description="Equation of state for a dilute ideal gas.",
        raw_formula="p V = n R T",
        variable_dimension_hints={
            "p": "M1L-1T-2",
            "V": "L3",
            "n": "N1",
            "R": "M1L2T-2Theta-1N-1",
            "T": "Theta1",
        },
        mechanism_tags=("equation_of_state", "molecular_kinetic_limit"),
        behavioral_archetypes=("state_relation", "linear_proportionality"),
        assumptions=("dilute gas", "negligible molecular interactions"),
        references=(
            {
                "title": "An Introduction to Thermal Physics",
                "author": "Daniel V. Schroeder",
                "year": "2000",
                "provenance_note": "Textbook reference for ideal-gas thermodynamics.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:thermodynamics:entropy_change_reversible",
        domain="thermodynamics",
        label="Reversible entropy change",
        description="Entropy differential for reversible heat transfer.",
        raw_formula="dS = delta Q_rev / T",
        variable_dimension_hints={
            "S": "M1L2T-2Theta-1",
            "Q_rev": "M1L2T-2",
            "T": "Theta1",
        },
        mechanism_tags=("entropy", "reversible_heat_transfer"),
        behavioral_archetypes=("state_function_differential", "thermodynamic_ratio"),
        assumptions=("reversible path", "well-defined thermodynamic temperature"),
        references=(
            {
                "title": "Thermodynamics and an Introduction to Thermostatistics",
                "author": "Herbert B. Callen",
                "year": "1985",
                "provenance_note": "Standard reference for entropy definitions.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:electromagnetism:coulomb_law",
        domain="electromagnetism",
        label="Coulomb's law",
        description="Electrostatic force between two point charges.",
        raw_formula="F = k_e q1 q2 / r^2",
        variable_dimension_hints={
            "F": "M1L1T-2",
            "k_e": "M1L3T-4I-2",
            "q1": "I1T1",
            "q2": "I1T1",
            "r": "L1",
        },
        mechanism_tags=("electrostatic_force", "inverse_square_law"),
        behavioral_archetypes=("inverse_square_decay", "long_range_interaction"),
        assumptions=("point charges", "static charges", "vacuum or stated medium"),
        references=(
            {
                "title": "Introduction to Electrodynamics",
                "author": "David J. Griffiths",
                "year": "2017",
                "provenance_note": "Textbook reference for electrostatics.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:electromagnetism:gauss_law",
        domain="electromagnetism",
        label="Gauss's law",
        description="Electric flux through a closed surface equals enclosed charge over permittivity.",
        raw_formula="oint E dot dA = Q_enc / epsilon_0",
        variable_dimension_hints={
            "E": "M1L1T-3I-1",
            "A": "L2",
            "Q_enc": "I1T1",
            "epsilon_0": "M-1L-3T4I2",
        },
        mechanism_tags=("field_flux", "charge_source", "maxwell_equation"),
        behavioral_archetypes=("integral_conservation_law", "source_field_relation"),
        assumptions=("closed surface", "classical electromagnetism"),
        references=(
            {
                "title": "A Treatise on Electricity and Magnetism",
                "author": "James Clerk Maxwell",
                "year": "1873",
                "provenance_note": "Historical source for Maxwellian field theory.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:electromagnetism:faraday_law",
        domain="electromagnetism",
        label="Faraday's law of induction",
        description="Circulating electric field is induced by changing magnetic flux.",
        raw_formula="oint E dot dl = - d Phi_B / dt",
        variable_dimension_hints={
            "E": "M1L1T-3I-1",
            "l": "L1",
            "Phi_B": "M1L2T-2I-1",
            "t": "T1",
        },
        mechanism_tags=("electromagnetic_induction", "flux_change", "maxwell_equation"),
        behavioral_archetypes=("time_derivative_drive", "circulation_law"),
        assumptions=("classical fields", "oriented contour and surface"),
        references=(
            {
                "title": "Experimental Researches in Electricity",
                "author": "Michael Faraday",
                "year": "1839",
                "provenance_note": "Historical source for induction phenomena.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:waves:wave_equation",
        domain="waves",
        label="Classical wave equation",
        description="Second time derivative balances spatial curvature for nondispersive waves.",
        raw_formula="partial^2 u / partial t^2 = c^2 nabla^2 u",
        variable_dimension_hints={"u": "problem_dependent", "t": "T1", "c": "L1T-1"},
        mechanism_tags=("wave_propagation", "spatial_curvature", "restoring_coupling"),
        behavioral_archetypes=("second_order_pde", "finite_speed_propagation"),
        assumptions=("linear medium", "constant wave speed", "nondispersive limit"),
        references=(
            {
                "title": "The Physics of Waves",
                "author": "Howard Georgi",
                "year": "1993",
                "provenance_note": "Textbook reference for wave equations.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:waves:dispersion_relation",
        domain="waves",
        label="Nondispersive dispersion relation",
        description="Angular frequency is proportional to wavenumber for fixed wave speed.",
        raw_formula="omega = c k",
        variable_dimension_hints={"omega": "T-1", "c": "L1T-1", "k": "L-1"},
        mechanism_tags=("dispersion_relation", "wave_speed"),
        behavioral_archetypes=("linear_proportionality", "frequency_wavenumber_relation"),
        assumptions=("nondispersive medium", "single propagation speed"),
        references=(
            {
                "title": "Vibrations and Waves",
                "author": "A. P. French",
                "year": "1971",
                "provenance_note": "Textbook reference for wave kinematics.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:waves:snell_law",
        domain="waves",
        label="Snell's law",
        description="Refraction conserves tangential phase across an interface.",
        raw_formula="n1 sin(theta1) = n2 sin(theta2)",
        variable_dimension_hints={
            "n1": "dimensionless",
            "theta1": "dimensionless",
            "n2": "dimensionless",
            "theta2": "dimensionless",
        },
        mechanism_tags=("refraction", "phase_matching", "interface_boundary_condition"),
        behavioral_archetypes=("boundary_condition", "dimensionless_relation"),
        assumptions=("geometric optics", "homogeneous media near interface"),
        references=(
            {
                "title": "Principles of Optics",
                "author": "Max Born; Emil Wolf",
                "year": "1999",
                "provenance_note": "Standard optics reference.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:diffusion_transport:fick_first_law",
        domain="diffusion_transport",
        label="Fick's first law",
        description="Diffusive flux is proportional to the negative concentration gradient.",
        raw_formula="J = -D nabla c",
        variable_dimension_hints={"J": "N1L-2T-1", "D": "L2T-1", "c": "N1L-3"},
        mechanism_tags=("diffusion", "gradient_driven_flux"),
        behavioral_archetypes=("linear_flux_law", "gradient_descent_transport"),
        assumptions=("constant diffusivity", "near-equilibrium linear response"),
        references=(
            {
                "title": "On Liquid Diffusion",
                "author": "Adolf Fick",
                "year": "1855",
                "provenance_note": "Historical source for Fickian diffusion.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:diffusion_transport:heat_equation",
        domain="diffusion_transport",
        label="Heat equation",
        description="Temperature evolves by thermal diffusion.",
        raw_formula="partial T / partial t = alpha nabla^2 T",
        variable_dimension_hints={"T": "Theta1", "t": "T1", "alpha": "L2T-1"},
        mechanism_tags=("thermal_diffusion", "laplacian_smoothing"),
        behavioral_archetypes=("parabolic_pde", "smoothing_dynamics"),
        assumptions=("constant thermal diffusivity", "isotropic homogeneous medium"),
        references=(
            {
                "title": "The Analytical Theory of Heat",
                "author": "Jean-Baptiste Joseph Fourier",
                "year": "1822",
                "provenance_note": "Historical source for heat conduction theory.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:diffusion_transport:ohm_law",
        domain="diffusion_transport",
        label="Ohm's law",
        description="Voltage drop is proportional to electric current.",
        raw_formula="V = I R",
        variable_dimension_hints={"V": "M1L2T-3I-1", "I": "I1", "R": "M1L2T-3I-2"},
        mechanism_tags=("linear_transport", "resistive_dissipation"),
        behavioral_archetypes=("linear_proportionality", "constitutive_relation"),
        assumptions=("ohmic material", "fixed temperature and geometry"),
        references=(
            {
                "title": "The Galvanic Circuit Investigated Mathematically",
                "author": "Georg Simon Ohm",
                "year": "1827",
                "provenance_note": "Historical source for Ohmic conduction.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:scaling:reynolds_number",
        domain="scaling_laws",
        label="Reynolds number",
        description="Dimensionless ratio of inertial to viscous effects in flow.",
        raw_formula="Re = rho v L / mu",
        variable_dimension_hints={
            "Re": "dimensionless",
            "rho": "M1L-3",
            "v": "L1T-1",
            "L": "L1",
            "mu": "M1L-1T-1",
        },
        mechanism_tags=("dimensional_analysis", "fluid_inertia", "viscous_dissipation"),
        behavioral_archetypes=("dimensionless_group", "regime_classifier"),
        assumptions=("continuum flow", "characteristic length and speed specified"),
        references=(
            {
                "title": "An Experimental Investigation of the Circumstances Which Determine Whether the Motion of Water Shall Be Direct or Sinuous",
                "author": "Osborne Reynolds",
                "year": "1883",
                "provenance_note": "Historical source for Reynolds-number scaling.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:scaling:stefan_boltzmann_law",
        domain="scaling_laws",
        label="Stefan-Boltzmann law",
        description="Blackbody radiated power per area scales with the fourth power of temperature.",
        raw_formula="j_star = sigma T^4",
        variable_dimension_hints={
            "j_star": "M1T-3",
            "sigma": "M1T-3Theta-4",
            "T": "Theta1",
        },
        mechanism_tags=("thermal_radiation", "power_law_scaling", "blackbody_limit"),
        behavioral_archetypes=("fourth_power_law", "scaling_law"),
        assumptions=("blackbody emitter", "thermal equilibrium"),
        references=(
            {
                "title": "The Theory of Heat Radiation",
                "author": "Max Planck",
                "year": "1914",
                "provenance_note": "Classical radiation theory reference.",
            },
        ),
    ),
    FoundationalLawSeed(
        source_id="manual:scaling:kepler_third_law",
        domain="scaling_laws",
        label="Kepler's third law",
        description="Orbital period squared scales with semi-major axis cubed.",
        raw_formula="T^2 = 4 pi^2 a^3 / (G M)",
        variable_dimension_hints={
            "T": "T1",
            "a": "L1",
            "G": "M-1L3T-2",
            "M": "M1",
        },
        mechanism_tags=("orbital_mechanics", "central_force", "power_law_scaling"),
        behavioral_archetypes=("power_law_relation", "period_length_scaling"),
        assumptions=("two-body problem", "central mass dominates", "Newtonian gravity"),
        references=(
            {
                "title": "Harmonices Mundi",
                "author": "Johannes Kepler",
                "year": "1619",
                "provenance_note": "Historical source for orbital period scaling.",
            },
        ),
    ),
)


def build_foundational_physics_backfill_bundle(
    *,
    source_version: str = SOURCE_VERSION,
    source_uri: str = SOURCE_URI,
    retrieved_at: str | None = None,
    snapshot_id: str | None = None,
) -> SourceAdapterBundle:
    """Build deterministic manual snapshot and candidate rows."""

    seeds = FOUNDATIONAL_LAW_SEEDS
    payload = {
        "source_kind": "curated_foundational_physics_backfill",
        "source_version": source_version,
        "record_count": len(seeds),
        "domains": sorted({seed.domain for seed in seeds}),
        "domain_counts": {
            domain: sum(1 for seed in seeds if seed.domain == domain)
            for domain in sorted({seed.domain for seed in seeds})
        },
        "records": [seed.to_payload(source_uri=source_uri) for seed in seeds],
    }
    return SourceAdapterBundle(
        snapshot_row=build_snapshot_row(
            source_system=SOURCE_SYSTEM,
            source_version=source_version,
            source_uri=source_uri,
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            payload=payload,
            license_expression=LICENSE_EXPRESSION,
            provenance_summary=PROVENANCE_SUMMARY,
            retrieved_at=retrieved_at,
        ),
        candidate_rows=tuple(
            seed.to_candidate_row(source_uri=source_uri, snapshot_id=snapshot_id)
            for seed in seeds
        ),
    )


def foundational_seed_domains() -> tuple[str, ...]:
    """Return deterministic domain coverage labels for the curated seed set."""

    return tuple(sorted({seed.domain for seed in FOUNDATIONAL_LAW_SEEDS}))
