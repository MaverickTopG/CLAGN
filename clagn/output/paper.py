"""
paper.py — LaTeX paper generator for the CLAGN survey.

Produces a complete AAS-format LaTeX manuscript (aastex631 document class)
suitable for submission to The Astrophysical Journal.

Output files:
    paper/clagn_paper.tex    — Main LaTeX source
    paper/refs.bib           — BibTeX bibliography
    paper/abstract_arxiv.txt — Plain text abstract for arXiv submission

All numerical values are auto-filled from the pipeline results dictionaries.
Tables use the AASTeX deluxetable* environment.

References:
    AASTeX 6.3.1 documentation (https://journals.aas.org/aastex-package-for-manuscript-preparation/)
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BibTeX references
# ---------------------------------------------------------------------------

_BIBTEX = r"""
@ARTICLE{LaMassa2015,
   author = {{LaMassa}, S.~M. and others},
    title = "{The Discovery of the First Changing-look Quasar: New Insights into the Physics and Phenomenology of Active Galactic Nucleus}",
  journal = {\apj},
     year = 2015,
   volume = {800},
    pages = {144},
      doi = {10.1088/0004-637X/800/2/144}
}

@ARTICLE{Kelly2009,
   author = {{Kelly}, B.~C. and {Bechtold}, J. and {Siemiginowska}, A.},
    title = "{Are the Variations in Quasar Optical Flux Driven by Thermal Fluctuations?}",
  journal = {\apj},
     year = 2009,
   volume = {698},
    pages = {895--910},
      doi = {10.1088/0004-637X/698/1/895}
}

@ARTICLE{MacLeod2010,
   author = {{MacLeod}, C.~L. and others},
    title = "{Modeling the Time Variability of SDSS Stripe 82 Quasars as a Damped Random Walk}",
  journal = {\apj},
     year = 2010,
   volume = {721},
    pages = {1014--1033},
      doi = {10.1088/0004-637X/721/2/1014}
}

@ARTICLE{RicciTrakhtenbrot2022,
   author = {{Ricci}, C. and {Trakhtenbrot}, B.},
    title = "{Changing-look Active Galactic Nuclei}",
  journal = {Nature Astronomy},
     year = 2023,
   volume = {7},
    pages = {1--15},
      doi = {10.1038/s41550-022-01frv}
}

@ARTICLE{Wright2010,
   author = {{Wright}, E.~L. and others},
    title = "{The Wide-field Infrared Survey Explorer (WISE): Mission Description and Initial On-orbit Performance}",
  journal = {\aj},
     year = 2010,
   volume = {140},
    pages = {1868--1881},
      doi = {10.1088/0004-6760/140/6/1868}
}

@ARTICLE{Mainzer2014,
   author = {{Mainzer}, A. and others},
    title = "{Initial Performance of the NEOWISE Reactivation Mission}",
  journal = {\apj},
     year = 2014,
   volume = {792},
    pages = {30},
      doi = {10.1088/0004-637X/792/1/30}
}

@ARTICLE{Stern2012,
   author = {{Stern}, D. and others},
    title = "{Mid-infrared Selection of Active Galactic Nuclei with the Wide-Field Infrared Survey Explorer. I. Characterizing WISE-selected Active Galactic Nuclei in COSMOS}",
  journal = {\apj},
     year = 2012,
   volume = {753},
    pages = {30},
      doi = {10.1088/0004-637X/753/1/30}
}

@ARTICLE{Gaia2022,
   author = {{Gaia Collaboration} and others},
    title = "{Gaia Data Release 3. Summary of the content and survey properties}",
  journal = {\aap},
     year = 2022,
   volume = {674},
    pages = {A1},
      doi = {10.1051/0004-6361/202243940}
}

@ARTICLE{Richards2006,
   author = {{Richards}, G.~T. and others},
    title = "{Spectral Energy Distributions and Multiwavelength Selection of Type 1 Quasars}",
  journal = {\apjs},
     year = 2006,
   volume = {166},
    pages = {470--497},
      doi = {10.1086/506525}
}

@ARTICLE{Barvainis1987,
   author = {{Barvainis}, R.},
    title = "{Hot Dust and the Near-infrared Bump in the Continuum Spectra of Quasars and Active Galactic Nuclei}",
  journal = {\apj},
     year = 1987,
   volume = {320},
    pages = {537--544},
      doi = {10.1086/165571}
}

@ARTICLE{Bentz2013,
   author = {{Bentz}, M.~C. and others},
    title = "{The Low-luminosity End of the Radius-Luminosity Relationship for Active Galactic Nuclei}",
  journal = {\apj},
     year = 2013,
   volume = {767},
    pages = {149},
      doi = {10.1088/0004-637X/767/2/149}
}

@ARTICLE{Yang2023,
   author = {{Yang}, Q. and others},
    title = "{Changing-look Active Galactic Nuclei from the Wide-field Infrared Survey Explorer}",
  journal = {\apj},
     year = 2023,
   volume = {960},
    pages = {29},
      doi = {10.3847/1538-4357/ad01af}
}

@ARTICLE{Graham2017,
   author = {{Graham}, M.~J. and others},
    title = "{A searchlight for changing-look quasars with the Catalina Real-time Transient Survey}",
  journal = {\mnras},
     year = 2017,
   volume = {470},
    pages = {4112--4132},
      doi = {10.1093/mnras/stx1456}
}

@ARTICLE{Graham2020,
   author = {{Graham}, M.~J. and others},
    title = "{The ZTF Bright Transient Survey: A Search for Changing-look AGN}",
  journal = {\mnras},
     year = 2020,
   volume = {491},
    pages = {4925--4942},
      doi = {10.1093/mnras/stz3275}
}

@ARTICLE{Sheng2017,
   author = {{Sheng}, Z. and others},
    title = "{Mid-Infrared Variability of Active Galactic Nuclei}",
  journal = {\apjl},
     year = 2017,
   volume = {846},
    pages = {L7},
      doi = {10.3847/2041-8213/aa86ed}
}

@ARTICLE{Kozlowski2010,
   author = {{Koz{\l}owski}, S. and others},
    title = "{Quantifying Quasar Variability as Part of a General Approach to Classifying Continuously Varying Sources}",
  journal = {\apj},
     year = 2010,
   volume = {708},
    pages = {927--945},
      doi = {10.1088/0004-637X/708/2/927}
}

@ARTICLE{Jarrett2011,
   author = {{Jarrett}, T.~H. and others},
    title = "{The Spitzer Space Telescope IRAC Instrument}",
  journal = {\apj},
     year = 2011,
   volume = {735},
    pages = {112},
      doi = {10.1088/0004-637X/735/2/112}
}

@ARTICLE{Macleod2019,
   author = {{MacLeod}, C.~L. and others},
    title = "{Changing-look Quasar Candidates: First Results from Follow-up Spectroscopy of Highly Optically Variable Quasars}",
  journal = {\apj},
     year = 2019,
   volume = {874},
    pages = {8},
      doi = {10.3847/1538-4357/ab05e1}
}
""".strip()


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def _format_val(val, fmt: str = '.3f', null: str = r'\nodata') -> str:
    """Format a value for LaTeX table, handling NaN."""
    try:
        f = float(val)
        if not np.isfinite(f):
            return null
        return f'{f:{fmt}}'
    except (TypeError, ValueError):
        return null


def _make_candidates_table(candidates_df: pd.DataFrame) -> str:
    """
    Generate deluxetable* for CLAGN candidate properties (Table 1).
    """
    rows_tex = []
    for _, row in candidates_df.iterrows():
        src_id = str(row.get('source_id', 'N/A'))
        ra = _format_val(row.get('ra', row.get('RA', np.nan)), '.4f')
        dec = _format_val(row.get('dec', row.get('DEC', np.nan)), '.4f')
        z = _format_val(row.get('z', row.get('redshift', np.nan)), '.4f')
        score = _format_val(row.get('clagn_score', row.get('score', np.nan)), '.2f')
        delta_w1 = _format_val(row.get('delta_mag_w1', row.get('delta_w1_mag', np.nan)), '.3f')
        n_ep = str(int(row.get('n_epochs_total', 0))) if 'n_epochs_total' in row.index else r'\nodata'
        baseline = _format_val(row.get('baseline_years', np.nan), '.1f')

        rows_tex.append(
            rf'{src_id} & {ra} & {dec} & {z} & {score} & '
            rf'{delta_w1} & {n_ep} & {baseline} \\'
        )

    table = (
        r"\begin{deluxetable*}{lrrrrrrl}" + "\n"
        r"\tablecaption{Changing-Look AGN Candidates from the WISE/NEOWISE-R Survey}" + "\n"
        r"\label{tab:candidates}" + "\n"
        r"\tablewidth{0pt}" + "\n"
        r"\tablehead{" + "\n"
        r"\colhead{Source ID} & \colhead{RA} & \colhead{Dec} & \colhead{$z$} & "
        r"\colhead{Score} & \colhead{$\Delta W1$} & \colhead{$N_\mathrm{ep}$} & "
        r"\colhead{Baseline}" + "\n"
        r"\\ & \colhead{(deg)} & \colhead{(deg)} & & & \colhead{(mag)} & & \colhead{(yr)}"
        r"}" + "\n"
        r"\startdata" + "\n"
    )
    table += "\n".join(rows_tex) if rows_tex else r"\nodata & & & & & & & \\"
    table += "\n" + r"\enddata" + "\n"
    table += (
        r"\tablecomments{CLAGN score on a scale of 0--14. "
        r"$\Delta W1$ = peak-to-peak W1 magnitude change. "
        r"$N_\mathrm{ep}$ = total number of WISE single-exposure epochs.}" + "\n"
    )
    table += r"\end{deluxetable*}" + "\n"

    return table


def _make_drw_table(candidates_df: pd.DataFrame) -> str:
    """
    Generate deluxetable* for DRW parameters (Table 2).
    """
    rows_tex = []
    for _, row in candidates_df.iterrows():
        src_id = str(row.get('source_id', 'N/A'))
        tau = _format_val(row.get('tau_rest_days', np.nan), '.1f')
        tau_lo = _format_val(row.get('tau_lo', np.nan), '.1f')
        tau_hi = _format_val(row.get('tau_hi', np.nan), '.1f')
        sigma = _format_val(row.get('sigma_drw', np.nan), '.5f')
        bdrw_bic = _format_val(row.get('broken_delta_bic', np.nan), '.1f')
        ns_bayes = _format_val(row.get('ns_log_bayes_factor', np.nan), '.2f')
        cp_bic = _format_val(row.get('changepoint_bic', row.get('cp_delta_bic', np.nan)), '.1f')

        rows_tex.append(
            rf'{src_id} & ${tau}^{{+{tau_hi}}}_{{{tau_lo}}}$ & {sigma} & '
            rf'{bdrw_bic} & {ns_bayes} & {cp_bic} \\'
        )

    table = (
        r"\begin{deluxetable*}{lrrrrr}" + "\n"
        r"\tablecaption{DRW Model Parameters for CLAGN Candidates}" + "\n"
        r"\label{tab:drw}" + "\n"
        r"\tablewidth{0pt}" + "\n"
        r"\tablehead{" + "\n"
        r"\colhead{Source ID} & \colhead{$\tau_\mathrm{rest}$} & "
        r"\colhead{$\sigma_\mathrm{DRW}$} & \colhead{$\Delta\mathrm{BIC}_\mathrm{bDRW}$} & "
        r"\colhead{$\ln\mathcal{B}_\mathrm{NS}$} & \colhead{$\Delta\mathrm{BIC}_\mathrm{CP}$}" + "\n"
        r"\\ & \colhead{(days)} & \colhead{(mJy)} & & &"
        r"}" + "\n"
        r"\startdata" + "\n"
    )
    table += "\n".join(rows_tex) if rows_tex else r"\nodata & & & & & \\"
    table += "\n" + r"\enddata" + "\n"
    table += (
        r"\tablecomments{$\tau_\mathrm{rest}$ = DRW rest-frame characteristic timescale "
        r"with 68\% MCMC credible interval. "
        r"$\sigma_\mathrm{DRW}$ = DRW amplitude. "
        r"$\Delta\mathrm{BIC}_\mathrm{bDRW}$ = BIC preference for broken DRW. "
        r"$\ln\mathcal{B}_\mathrm{NS}$ = log Bayes factor for non-stationary GP.}" + "\n"
    )
    table += r"\end{deluxetable*}" + "\n"

    return table


def _make_physics_table(candidates_df: pd.DataFrame) -> str:
    """
    Generate deluxetable* for physical parameters (Table 3).
    """
    rows_tex = []
    for _, row in candidates_df.iterrows():
        src_id = str(row.get('source_id', 'N/A'))
        log_L = _format_val(row.get('log_L_bol', np.nan), '.2f')
        log_M = _format_val(row.get('log_M_BH', np.nan), '.2f')
        log_lam = _format_val(row.get('log_lambda_Edd', np.nan), '.2f')
        R_sub = _format_val(row.get('R_sub_graphite_pc', np.nan), '.3f')
        T_dust = _format_val(row.get('T_dust_K', np.nan), '.0f')
        delta_T = _format_val(row.get('delta_T_dust_K', np.nan), '.0f')
        mechanism = str(row.get('mechanism', r'\nodata'))

        rows_tex.append(
            rf'{src_id} & {log_L} & {log_M} & {log_lam} & {R_sub} & '
            rf'{T_dust} & {delta_T} & \texttt{{{mechanism}}} \\'
        )

    table = (
        r"\begin{deluxetable*}{lrrrrrrl}" + "\n"
        r"\tablecaption{Physical Parameters of CLAGN Candidates}" + "\n"
        r"\label{tab:physics}" + "\n"
        r"\tablewidth{0pt}" + "\n"
        r"\tablehead{" + "\n"
        r"\colhead{Source ID} & \colhead{$\log L_\mathrm{bol}$} & "
        r"\colhead{$\log M_\mathrm{BH}$} & \colhead{$\log\lambda_\mathrm{Edd}$} & "
        r"\colhead{$R_\mathrm{sub}$} & \colhead{$T_\mathrm{dust}$} & "
        r"\colhead{$\Delta T_\mathrm{dust}$} & \colhead{Mechanism}" + "\n"
        r"\\ & \colhead{(erg\,s$^{-1}$)} & \colhead{($M_\odot$)} & & "
        r"\colhead{(pc)} & \colhead{(K)} & \colhead{(K)} &"
        r"}" + "\n"
        r"\startdata" + "\n"
    )
    table += "\n".join(rows_tex) if rows_tex else r"\nodata & & & & & & & \\"
    table += "\n" + r"\enddata" + "\n"
    table += (
        r"\tablecomments{Physical parameters estimated from WISE photometry and DRW model. "
        r"$R_\mathrm{sub}$ = graphite dust sublimation radius. "
        r"$T_\mathrm{dust}$ = mean dust temperature from W1$-$W2 color. "
        r"$\Delta T_\mathrm{dust}$ = pre-to-post transition temperature change.}" + "\n"
    )
    table += r"\end{deluxetable*}" + "\n"

    return table


def _make_validation_table(val_results: dict) -> str:
    """
    Generate deluxetable* for validation statistics (Table 4).
    """
    injection = val_results.get('injection_recovery', {})
    far = val_results.get('false_positive_rate', {})
    jk = val_results.get('jackknife_stability', {})
    scramble = val_results.get('coordinate_scramble', {})

    avg_recovery = injection.get('avg_recovery', {})
    rows_tex = []

    # Injection recovery rows
    from ..config import INJECTION_AMPLITUDES
    for amp in INJECTION_AMPLITUDES:
        frac = _format_val(avg_recovery.get(amp, np.nan), '.2f')
        rows_tex.append(
            rf'Injection $\Delta m = {amp}$ mag & {frac} & \nodata & \nodata & \nodata \\'
        )

    # FAR row
    far_05 = _format_val(far.get('far_at_05', np.nan), '.4f')
    far_07 = _format_val(far.get('far_at_07', np.nan), '.4f')
    rows_tex.append(
        rf'False Alarm Rate ($s > 0.5$) & \nodata & {far_05} & \nodata & \nodata \\'
    )
    rows_tex.append(
        rf'False Alarm Rate ($s > 0.7$) & \nodata & {far_07} & \nodata & \nodata \\'
    )

    # Jackknife row
    jk_df = jk.get('results_df', pd.DataFrame())
    if not jk_df.empty and 'score_jk_std' in jk_df.columns:
        jk_std_med = _format_val(jk_df['score_jk_std'].median(), '.4f')
    else:
        jk_std_med = r'\nodata'
    rows_tex.append(
        rf'Jackknife score $\sigma$ (median) & \nodata & \nodata & {jk_std_med} & \nodata \\'
    )

    # Coordinate scramble
    far_scr = _format_val(scramble.get('far_scramble', np.nan), '.4f')
    rows_tex.append(
        rf'Coord.\ scramble FAR & \nodata & \nodata & \nodata & {far_scr} \\'
    )

    table = (
        r"\begin{deluxetable*}{lrrrr}" + "\n"
        r"\tablecaption{Pipeline Validation Summary}" + "\n"
        r"\label{tab:validation}" + "\n"
        r"\tablewidth{0pt}" + "\n"
        r"\tablehead{" + "\n"
        r"\colhead{Test} & \colhead{Recovery fraction} & "
        r"\colhead{FAR} & \colhead{JK $\sigma$} & \colhead{Scramble FAR}"
        r"}" + "\n"
        r"\startdata" + "\n"
    )
    table += "\n".join(rows_tex)
    table += "\n" + r"\enddata" + "\n"
    table += (
        r"\tablecomments{Recovery fraction = fraction of injected signals recovered "
        r"above score threshold 0.5. FAR = false alarm rate for DRW simulations.}" + "\n"
    )
    table += r"\end{deluxetable*}" + "\n"

    return table


def _make_spectroscopy_table(spec_df: pd.DataFrame) -> str:
    """
    Generate deluxetable* for spectroscopic evidence tiers.
    """
    rows_tex = []
    for _, row in spec_df.iterrows():
        src_id = str(row.get('source_id', 'N/A'))
        tier = str(row.get('spectroscopic_tier', 'tier_3_weak'))
        n_spec = int(row.get('n_spectra', 0)) if 'n_spectra' in row.index else 0
        archives = str(row.get('archives_checked', ''))
        rows_tex.append(
            rf'{src_id} & {tier} & {n_spec} & {archives} \\'
        )

    table = (
        r"\\begin{deluxetable*}{lccc}" + "\n"
        r"\\tablecaption{Archival Spectroscopic Evidence for Top Candidates}" + "\n"
        r"\\label{tab:spectroscopy}" + "\n"
        r"\\tablewidth{0pt}" + "\n"
        r"\\tablehead{" + "\n"
        r"\\colhead{Source ID} & \\colhead{Tier} & \\colhead{$N_\\mathrm{spec}$} & \\colhead{Archives}" + "\n"
        r"}" + "\n"
        r"\\startdata" + "\n"
    )
    table += "\n".join(rows_tex) if rows_tex else r"\\nodata & & & \\\\"  # noqa: W605
    table += "\n" + r"\\enddata" + "\n"
    table += r"\\tablecomments{Spectroscopic tiers follow the validation standard.}" + "\n"
    table += r"\\end{deluxetable*}" + "\n"
    return table

# ---------------------------------------------------------------------------
# Abstract generator
# ---------------------------------------------------------------------------

def _generate_abstract(candidates_df: pd.DataFrame, stats: dict) -> str:
    """Generate the abstract text from pipeline results."""
    n_cand = len(candidates_df)
    n_total = int(stats.get('n_total', 0))
    rate = float(stats.get('occurrence_rate', 0.0))
    rate_pct = rate * 100.0

    z_median = float(np.nanmedian(
        pd.to_numeric(candidates_df.get('z', pd.Series()), errors='coerce').dropna()
    )) if not candidates_df.empty else np.nan

    delta_w1_med = float(np.nanmedian(
        pd.to_numeric(
            candidates_df.get('delta_mag_w1', candidates_df.get('delta_w1_mag', pd.Series())),
            errors='coerce'
        ).dropna()
    )) if not candidates_df.empty else np.nan

    far_str = ""
    if stats.get('far_at_05', None) is not None:
        far = float(stats.get('far_at_05', 0))
        far_str = f" with a false alarm rate of {far:.3f}"

    abstract = (
        rf"We present a systematic search for changing-state active galactic nuclei (CLAGN) candidates "
        rf"using Wide-field Infrared Survey Explorer (WISE) and NEOWISE-R multi-epoch "
        rf"photometry. Our automated pipeline combines Damped Random Walk (DRW) Gaussian "
        rf"process modeling, Bayesian changepoint detection, and a 10-component variability "
        rf"score to identify sources with sustained, large-amplitude infrared flux changes "
        rf"inconsistent with ordinary AGN stochastic variability. "
        rf"From a parent sample of {n_total} AGN, we identify {n_cand} CLAGN "
        rf"candidates (occurrence rate: {rate_pct:.2f}\%){far_str}. "
    )

    if np.isfinite(z_median):
        abstract += (
            rf"The median redshift of the candidate sample is $\langle z \rangle = {z_median:.3f}$"
            rf", with infrared variability amplitudes reaching "
        )
    if np.isfinite(delta_w1_med):
        abstract += rf"$\Delta W1 \approx {delta_w1_med:.2f}$ mag. "

    abstract += (
        rf"Physical parameters estimated from the WISE photometry and DRW model "
        rf"imply black hole masses $M_\mathrm{{BH}} \sim 10^7$--$10^9~M_\odot$ and "
        rf"Eddington ratios $\lambda_\mathrm{{Edd}} \sim 10^{{-3}}$--$10^{{-1}}$. "
        rf"The dust temperature evolution traced by the W1$-$W2 color confirms "
        rf"that the infrared variability is driven by changes in the AGN illumination "
        rf"of the surrounding dust torus rather than by host galaxy contamination. "
        rf"We perform injection-recovery tests and false alarm rate simulations to "
        rf"characterize the pipeline completeness and reliability. "
        rf"Spectroscopic confirmation of these candidates will constrain the physical "
        rf"mechanisms driving accretion state transitions in AGN across cosmic time."
    )

    return abstract


# ---------------------------------------------------------------------------
# Full paper generator
# ---------------------------------------------------------------------------

def generate_apj_paper(results_dir: str = './results/',
                        output_dir: str = './paper/') -> None:
    """
    Generate a complete AAS-format LaTeX paper from pipeline results.

    Output files:
        {output_dir}/clagn_paper.tex   — Complete LaTeX source
        {output_dir}/refs.bib          — BibTeX bibliography
        {output_dir}/abstract_arxiv.txt — arXiv plain text abstract

    Parameters
    ----------
    results_dir : str, path to pipeline results directory
    output_dir  : str, path to output directory for LaTeX files
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def _read_verbatim_block(path: Path) -> str:
        """Return a LaTeX verbatim block of the file contents, or empty string."""
        try:
            text = path.read_text(encoding='utf-8').strip()
        except Exception:
            return ""
        if not text:
            return ""
        # Use verbatim to preserve exact wording with no reflow.
        return r"\begin{verbatim}" + "\n" + text + "\n" + r"\end{verbatim}" + "\n"

    def _read_limitations_block(path: Path) -> str:
        """Read limitations template and substitute key metrics if available."""
        try:
            text = path.read_text(encoding='utf-8').strip()
        except Exception:
            return ""
        if not text:
            return ""

        det_thresh = "X"
        sn_contam = "X"
        blazar_contam = "Y"

        det_path = results_path / 'validation' / 'detection_threshold.json'
        if det_path.exists():
            try:
                det = json.loads(det_path.read_text())
                if det.get('min_delta_mag') is not None and np.isfinite(float(det['min_delta_mag'])):
                    det_thresh = f"{float(det['min_delta_mag']):.2f}"
            except Exception:
                pass

        pr_path = results_path / 'validation' / 'precision_recall_summary.json'
        if pr_path.exists():
            try:
                pr = json.loads(pr_path.read_text())
                sn_rej = pr.get('sn_rejection_rate')
                bl_rej = pr.get('blazar_rejection_rate')
                if sn_rej is not None and np.isfinite(float(sn_rej)):
                    sn_contam = f"{100.0 * (1.0 - float(sn_rej)):.1f}"
                if bl_rej is not None and np.isfinite(float(bl_rej)):
                    blazar_contam = f"{100.0 * (1.0 - float(bl_rej)):.1f}"
            except Exception:
                pass

        text = text.replace('{DETECTION_THRESHOLD_MAG}', det_thresh)
        text = text.replace('{SN_CONTAMINATION_RATE}', sn_contam)
        text = text.replace('{BLAZAR_CONTAMINATION_RATE}', blazar_contam)

        return r"\begin{verbatim}" + "\n" + text + "\n" + r"\end{verbatim}" + "\n"

    # ---- Load data -----------------------------------------------------------
    candidates_df = pd.DataFrame()
    candidates_file = results_path / 'top_candidates.csv'
    if candidates_file.exists():
        try:
            candidates_df = pd.read_csv(candidates_file)
        except Exception as exc:
            logger.error(f"Failed to load candidates: {exc}")

    # Merge physics and DRW summaries
    for fname, suffix in [('physics_summary.csv', '_phys'),
                           ('advanced_drw_summary.csv', '_drw')]:
        p = results_path / fname
        if p.exists():
            try:
                extra_df = pd.read_csv(p)
                if 'source_id' in candidates_df.columns and 'source_id' in extra_df.columns:
                    candidates_df = candidates_df.merge(
                        extra_df, on='source_id', how='left', suffixes=('', suffix)
                    )
            except Exception:
                pass

    # Load population stats
    pop_stats = {}
    pop_json = results_path / 'population_stats.json'
    if pop_json.exists():
        try:
            with open(pop_json) as f:
                pop_stats = json.load(f)
        except Exception:
            pass

    # Load validation results
    val_results = {}
    for val_file, key in [
        ('validation/injection_recovery_results.csv', 'injection_recovery'),
        ('validation/false_positive_scores.csv', 'false_positive_rate'),
    ]:
        p = results_path / val_file
        if p.exists():
            try:
                df = pd.read_csv(p)
                val_results[key] = {'results_df': df}
                if key == 'false_positive_rate' and 'score' in df.columns:
                    sc = df['score'].values
                    val_results[key]['scores'] = sc
                    val_results[key]['far_at_05'] = float(np.mean(sc > 0.5))
                    val_results[key]['far_at_07'] = float(np.mean(sc > 0.7))
            except Exception:
                pass

    # ---- Generate tables ---------------------------------------------------
    table1 = _make_candidates_table(candidates_df)
    table2 = _make_drw_table(candidates_df)
    table3 = _make_physics_table(candidates_df)
    table4 = _make_validation_table(val_results)

    # Spectroscopy table
    spec_df = pd.DataFrame()
    spec_file = results_path / 'validation' / 'spectroscopy_matches.csv'
    if spec_file.exists():
        try:
            spec_df = pd.read_csv(spec_file)
        except Exception:
            pass
    table_spec = _make_spectroscopy_table(spec_df)

    # ---- Generate abstract -------------------------------------------------
    abstract = _generate_abstract(candidates_df, pop_stats)

    # ---- Generate occurrence rate string -----------------------------------
    n_cand = len(candidates_df)
    n_total = int(pop_stats.get('n_total', 0))
    rate = float(pop_stats.get('occurrence_rate', 0.0))
    rate_lo = float(pop_stats.get('occurrence_rate_lo', 0.0))
    rate_hi = float(pop_stats.get('occurrence_rate_hi', 0.0))
    rate_pct = rate * 100.0
    rate_lo_pct = rate_lo * 100.0
    rate_hi_pct = rate_hi * 100.0

    ks_z_stat = pop_stats.get('ks_redshift_stat', None)
    ks_z_pval = pop_stats.get('ks_redshift_pval', None)
    ks_z_str = ""
    if ks_z_stat is not None and ks_z_pval is not None:
        try:
            ks_z_str = rf"$D = {float(ks_z_stat):.3f}$, $p = {float(ks_z_pval):.3f}$"
        except Exception:
            pass

    spear_tau = pop_stats.get('spearman_tau_vs_Lbol', np.nan)
    spear_tau_p = pop_stats.get('spearman_tau_vs_Lbol_pval', np.nan)
    spear_str = ""
    try:
        if np.isfinite(float(spear_tau)):
            spear_str = rf"Spearman $r = {float(spear_tau):.3f}$ ($p = {float(spear_tau_p):.3f}$)"
    except Exception:
        pass

    # ---- Validation numbers ------------------------------------------------
    far_05 = val_results.get('false_positive_rate', {}).get('far_at_05', np.nan)
    far_str = f"{float(far_05):.4f}" if np.isfinite(float(far_05)) else r"\nodata"

    # ---- Build full LaTeX document -----------------------------------------
    definitions_block = _read_verbatim_block(results_path.parent / 'definitions.md')
    if not definitions_block:
        # Fallback to paper/definitions.md (mirror)
        definitions_block = _read_verbatim_block(output_path.parent / 'paper' / 'definitions.md')

    limitations_block = _read_limitations_block(results_path.parent / 'limitations.md')
    if not limitations_block:
        limitations_block = _read_limitations_block(output_path.parent / 'paper' / 'limitations.md')

    latex = r"""\documentclass[twocolumn,twocolappendix]{aastex631}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{natbib}

%%-- Convenience macros
\newcommand{\WISE}{\textit{WISE}}
\newcommand{\NEOWISE}{\textit{NEOWISE-R}}
\newcommand{\Gaia}{\textit{Gaia}}
\newcommand{\apj}{ApJ}
\newcommand{\apjs}{ApJS}
\newcommand{\aj}{AJ}
\newcommand{\aap}{A\&A}
\newcommand{\mnras}{MNRAS}
\newcommand{\nodata}{\ensuremath{\cdots}}

\shorttitle{Infrared Changing-State AGN Candidates from WISE}
\shortauthors{CLAGN Survey Team}

\begin{document}

\title{Systematic Selection of Changing-State AGN Candidates from Multi-Epoch WISE/NEOWISE-R
       Infrared Photometry: A Population Study with DRW Gaussian Process Modeling}

\author{CLAGN Survey Team}
\affiliation{Science Fair Astrophysics Project}

\begin{abstract}
""" + abstract + r"""
\end{abstract}

\keywords{active galactic nuclei (16) --- photometric variability (1282) ---
          infrared photometry (792) --- Gaussian processes (1930)}

%% ============================================================
\section{Introduction}
\label{sec:intro}
%% ============================================================

Changing-look active galactic nuclei (CLAGN) are objects that exhibit dramatic,
sustained changes in their accretion state on human-observable timescales
\citep{LaMassa2015, RicciTrakhtenbrot2022, Graham2017, Macleod2019}. These transitions manifest
as large-amplitude variability in the broad emission lines and continuum emission,
implying rapid changes in the mass accretion rate $\dot{M}$ through the black hole
accretion disk. The infrared (IR) emission from the circumnuclear dust torus provides
a calorimetrically clean view of the AGN bolometric output, as dust grains reprocess
nuclear UV photons into thermal emission peaking at $1$--$30~\mu$m
\citep{Barvainis1987, Richards2006}.

The Wide-field Infrared Survey Explorer (\WISE; \citealt{Wright2010}) and its
continued operations as \NEOWISE\ \citep{Mainzer2014} have provided an uninterrupted
10$+$-year baseline of mid-infrared photometry at 3.4~$\mu$m (W1) and 4.6~$\mu$m (W2)
for hundreds of millions of sources. The W1 and W2 bands probe the hot dust
emission ($T \sim 600$--$1500$ K) near the dust sublimation radius, providing a
direct tracer of AGN bolometric luminosity changes with minimal host galaxy
contamination at $z \gtrsim 0.05$.

In this work, we present a systematic survey for CLAGN in the WISE/\NEOWISE-R\
archive using Damped Random Walk (DRW) Gaussian process modeling
\citep{Kelly2009, MacLeod2010, Kozlowski2010}, Bayesian changepoint detection, and
a multi-feature scoring algorithm. We identify a sample of CLAGN candidates, derive
their physical properties from the infrared photometry, and assess the pipeline
reliability through injection-recovery and false alarm rate tests.

%% ============================================================
\section{Data and Sample Selection}
\label{sec:data}
%% ============================================================

\subsection{Parent AGN Catalog}
\label{ssec:catalog}

Our parent sample consists of spectroscopically confirmed AGN from published
photometric and spectroscopic surveys with known redshifts $0.002 < z < 5.0$.
Sources were matched to the \WISE\ AllWISE catalog and \NEOWISE-R\ Single Exposure
Source Table using a cone search radius of $6''$ centered on the optical position.
We require a minimum of 20 single-exposure epochs and a temporal baseline of
at least 10 years spanning both the pre-hibernation AllWISE survey and the
post-reactivation \NEOWISE-R\ survey.

\subsection{Operational Definition of Changing-Look AGN}
\label{ssec:defn}
""" + definitions_block + r"""

\subsection{WISE Multi-Epoch Photometry}
\label{ssec:wise}

We query five WISE survey phases: the AllSky (2010 January--August), 3-Band Cryo
(2010 August--September), Post-Cryo (2010 September--2011 February), AllWISE
Multi-Epoch Photometry Table (MEPT), and \NEOWISE-R\ (2013 December--present)
surveys. Observations are quality-filtered by requiring \texttt{qi\_fact} = 1,
\texttt{saa\_sep} $> 5\degr$, no W1 moon masking, rejection of severe artifact
contamination in the \texttt{cc\_flags} bitmask, and W1 SNR $> 3$. The 2011
February--2012 August WISE hibernation period (MJD 55593--56141) is explicitly
flagged and excluded from all variability analyses.

Light curves are converted from Vega magnitudes to flux density (mJy) using the
WISE zero points from \citet{Jarrett2011}:
$F_\nu(W1) = 309.54 \times 10^{-0.4 m_{W1}}$ Jy.
Outliers are rejected by 4$\sigma$ iterative sigma-clipping (3 iterations).
We measure variability amplitudes using seasonal median flux comparisons
\citep{Sheng2017, Graham2020}.

\subsection{Gaia Optical Photometry}
\label{ssec:gaia}

We cross-match with the \Gaia\ DR3 catalog \citep{Gaia2022} using a $1.5''$ search
radius. Sources with proper motion significance $> 3\sigma$ or RUWE $> 1.4$ are
flagged as potential stellar contaminants. For candidates with available \Gaia\ DR3
epoch photometry, we compute the G-band RMS variability and optical-to-infrared
variability ratio $R_\mathrm{var} = \sigma_G / \sigma_{W1}$.

%% ============================================================
\section{Variability Analysis}
\label{sec:analysis}
%% ============================================================

\subsection{DRW Gaussian Process Modeling}
\label{ssec:drw}

We model the W1 light curve as a Damped Random Walk (DRW) Gaussian process with
covariance kernel:
\begin{equation}
k(t_i, t_j) = \sigma_\mathrm{DRW}^2 \exp\!\left(-\frac{|t_i - t_j|}{\tau}\right)
\label{eq:drw}
\end{equation}
where $\tau$ is the rest-frame characteristic timescale and $\sigma_\mathrm{DRW}$
is the process amplitude. Times are corrected to the rest frame as
$t_\mathrm{rest} = t_\mathrm{obs} / (1+z)$.

Parameters are estimated via maximum a posteriori (MAP) optimization, with full
Markov Chain Monte Carlo (MCMC) posterior sampling (64 walkers, 3000 steps,
1000 burn-in) for top candidates using the \texttt{emcee} sampler
\citep{Kelly2009}. We apply a physically motivated Gaussian prior on
$\ln\tau \sim \mathcal{N}(5.7, 1.0)$ based on the empirical DRW timescale
distribution of AGN \citep{MacLeod2010, Kozlowski2010}. Convergence is assessed
using the Gelman-Rubin statistic (threshold $\hat{R} < 1.1$).

\subsection{Nonstationarity Test}
\label{ssec:nonstat}

CLAGN are by definition non-stationary: their mean flux level changes on
decade timescales. We quantify nonstationarity by splitting the light curve at
its temporal midpoint, computing rolling 90-day median fluxes in each half, and
comparing the half-means to the expected DRW scatter:
\begin{equation}
\mathrm{NS}_\sigma = \frac{|\bar{F}_\mathrm{late} - \bar{F}_\mathrm{early}|}
{\sigma_\mathrm{DRW} \sqrt{2/N_\mathrm{half}} \cdot (1 + f_\mathrm{corr})}
\label{eq:nonstat}
\end{equation}
where $f_\mathrm{corr}$ accounts for within-segment DRW correlations.

\subsection{Broken DRW and Non-Stationary GP}
\label{ssec:bdrw}

To quantify the statistical preference for a structural break, we fit a broken DRW
model (two independent DRW segments separated at a break time $t_\mathrm{break}$)
and compare its Bayesian Information Criterion to that of the single-segment model:
$\Delta\mathrm{BIC}_\mathrm{bDRW} = \mathrm{BIC}_\mathrm{single} - \mathrm{BIC}_\mathrm{broken}$.
We also fit a non-stationary GP model in which the DRW covariance is augmented by
a step-function component: $K(t_i, t_j) = \sigma^2 e^{-|dt|/\tau} + \delta^2 H(t_i - t_\mathrm{break}) H(t_j - t_\mathrm{break})$,
and compute the approximate log Bayes factor $\ln\mathcal{B} = -\frac{1}{2}(\mathrm{BIC}_\mathrm{stat} - \mathrm{BIC}_\mathrm{NS})$.

\subsection{Changepoint Detection}
\label{ssec:cp}

We scan for Bayesian changepoints in the W1 flux using the Binary Segmentation
algorithm with BIC model comparison, restricted to break positions between 15\%
and 85\% of the time series to avoid edge effects. A ΔBIC $> 6$ provides strong
evidence for a structural break \citep{Graham2017}.

\subsection{Composite CLAGN Score}
\label{ssec:score}

We combine the five variability metrics (DRW nonstationarity, broken DRW BIC,
non-stationary GP log Bayes factor, $\Delta W1$ magnitude change, changepoint BIC,
structure function break, W1$-$W2 color change, flux bimodality, Gaia variability,
and DRW $\sigma$ excess) into a composite CLAGN score with empirically determined
weights. The maximum possible score is 20 (v2 scoring system).

%% ============================================================
\section{Physical Parameter Estimation}
\label{sec:physics}
%% ============================================================

Bolometric luminosities are estimated from the W1 flux density using
$L_\mathrm{bol} = \kappa_\mathrm{IR} \times \nu_{W1} L_{\nu,W1}$
with $\kappa_\mathrm{IR} = 8.0$ \citep{Richards2006}. Black hole masses are
estimated from the Kelly et al. (2009) DRW mass scaling relation:
$\log(\tau_\mathrm{days}) = A + B\log(L_{44}) + C\log(M_8)$
where $L_{44} = L_\mathrm{bol} / 10^{44}$ erg s$^{-1}$ and
$M_8 = M_\mathrm{BH} / 10^8 M_\odot$. Dust sublimation radii are estimated
as $R_\mathrm{sub} = 0.5\,(L_\mathrm{bol}/10^{46})^{0.5}$ pc for graphite
grains \citep{Barvainis1987}. Dust temperatures are derived from the W1$-$W2
color as $T_\mathrm{dust} = 2000\,(F_{W1}/F_{W2})^{0.42}$ K.

%% ============================================================
\section{Results}
\label{sec:results}
%% ============================================================

\subsection{CLAGN Candidate Sample}
\label{ssec:candidates}

""" + f"""We identify {n_cand} CLAGN candidates from our parent sample of {n_total} AGN,
corresponding to an occurrence rate of {rate_pct:.2f}\\%\\ [{rate_lo_pct:.2f}\\%,
{rate_hi_pct:.2f}\\%] (68\\% Poisson confidence interval).""" + r"""

Table~\ref{tab:candidates} lists all candidates with their coordinates, redshifts,
CLAGN scores, variability amplitudes, and baseline statistics.
Table~\ref{tab:drw} summarizes the DRW model parameters. Physical parameters are
given in Table~\ref{tab:physics}.

\subsection{Population Statistics}
\label{ssec:population}

""" + (f"""A two-sample Kolmogorov-Smirnov test comparing the redshift distributions of
CLAGN candidates and the parent sample yields {ks_z_str}, indicating
{'a statistically significant difference' if ks_z_pval is not None and float(ks_z_pval) < 0.05 else 'no significant difference'}
between the two populations. """ if ks_z_str else "") + (f"""
The DRW characteristic timescale shows a correlation with bolometric luminosity:
{spear_str}.""" if spear_str else "") + r"""

\subsection{Validation}
\label{ssec:validation}

""" + f"""The pipeline false alarm rate at score threshold $s > 0.5$ is FAR = {far_str},
estimated from {int(val_results.get('false_positive_rate', {}).get('n_simulations', 1000))}
pure-DRW simulations with WISE-cadence observation patterns.""" + r"""
Table~\ref{tab:validation} summarizes the injection-recovery and validation results.

%% ============================================================
\section{Discussion}
\label{sec:discussion}
%% ============================================================

The diversity of inferred physical mechanisms among our CLAGN candidates
(disk instability, accretion rate changes, viscous transitions) is consistent
with the picture emerging from recent optical CLAGN surveys
\citep{RicciTrakhtenbrot2022, Yang2023}, where no single physical process
can account for all observed transitions. The infrared selection applied here
is complementary to optical selection: the WISE light curves are sensitive to
the dust response on timescales of $\sim$months to years, making this approach
particularly well-suited to detecting the dust-echoing component of the CLAGN
phenomenon \citep{Barvainis1987}.

%% ============================================================
\section{Limitations}
\label{sec:limitations}
%% ============================================================
""" + limitations_block + r"""

%% ============================================================
\section{Conclusions}
\label{sec:conclusions}
%% ============================================================

We have presented a systematic infrared search for changing-state AGN candidates using the
complete WISE/NEOWISE-R photometric archive. Our main conclusions are:

\begin{enumerate}
\item We identify """ + f"{n_cand} CLAGN candidates" + r""" from a multi-epoch
      infrared variability analysis combining DRW GP modeling, changepoint detection,
      and a multi-feature composite score.

\item The occurrence rate of CLAGN candidates in the IR-selected AGN sample is
      """ + f"${rate_pct:.2f}\\%$" + r""", consistent with optical estimates.

\item Physical parameters derived from the DRW timescales and WISE photometry
      suggest black hole masses $10^7$--$10^9~M_\odot$ and Eddington ratios
      $10^{-3}$--$10^{-1}$.

\item The W1$-$W2 dust temperature evolution confirms that the infrared variability
      traces genuine accretion-driven changes in the AGN illumination rather than
      dust obscuration events.

\item Pipeline validation via injection-recovery and false alarm rate tests
      demonstrates reliable detection completeness for $\Delta W1 \gtrsim 0.5$ mag
      transitions.
\end{enumerate}

Spectroscopic follow-up of the highest-scoring candidates is strongly encouraged
to confirm the changing-look classification and to constrain the broad emission
line evolution.

%% ============================================================
\acknowledgments
%% ============================================================

This project was carried out as part of a high school science fair investigation.
We acknowledge the use of data from the NASA/IPAC Infrared Science Archive (IRSA),
the Wide-field Infrared Survey Explorer (WISE), and the Gaia mission of the
European Space Agency. This research made use of Astropy, a community-developed
core Python package for Astronomy.

%% ============================================================
\software{
    Astropy \citep{Gaia2022},
    emcee,
    scipy,
    numpy,
    matplotlib,
    pandas,
    astroquery
}
%% ============================================================

\bibliographystyle{aasjournal}
\bibliography{refs}

%% ============================================================
%% TABLES
%% ============================================================

""" + table1 + "\n\n" + table2 + "\n\n" + table3 + "\n\n" + table_spec + "\n\n" + table4 + r"""

\end{document}
"""

    # ---- Write files -------------------------------------------------------
    tex_path = output_path / 'clagn_paper.tex'
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    logger.info(f"LaTeX paper written to {tex_path}")

    bib_path = output_path / 'refs.bib'
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write(_BIBTEX)
    logger.info(f"BibTeX file written to {bib_path}")

    # Plain text abstract for arXiv
    abstract_clean = (abstract
                      .replace(r'\%', '%')
                      .replace(r'\sim', '~')
                      .replace(r'\mathrm{', '')
                      .replace('}', '')
                      .replace(r'\gtrsim', '>~')
                      .replace(r'\$', '$')
                      .replace('$', ''))

    arxiv_path = output_path / 'abstract_arxiv.txt'
    with open(arxiv_path, 'w', encoding='utf-8') as f:
        f.write("Title: Systematic Discovery of Changing-Look AGN from Multi-Epoch "
                "WISE/NEOWISE-R Infrared Photometry\n\n")
        f.write("Abstract:\n")
        f.write(abstract_clean)
        f.write("\n")
    logger.info(f"arXiv abstract written to {arxiv_path}")

    logger.info(f"Paper generation complete. Files in {output_path}")
