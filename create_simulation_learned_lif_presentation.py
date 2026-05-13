from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
import torch


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PPTX = BASE_DIR / "LIF_Simulation_Learned_LIF_Presentation.pptx"
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

ASSETS = {
    "bursts": BASE_DIR / "network spikes (resampled ata 20Hz).png",
    "raw_raster": BASE_DIR / "raster_plot.png",
    "voltage_traces": BASE_DIR / "output.png",
    "metrics": BASE_DIR / "presentation_assets" / "recent_metrics.png",
    "tradeoff": BASE_DIR / "presentation_assets" / "candidate_tradeoff.png",
    "connectivity": BASE_DIR / "learned_lif_outputs" / "learned_lif_20260423_091632_visible_lines.png",
}

CONNECTIVITY_DATASETS = [
    {
        "label": "0423 dataset",
        "session": "20260423_091632",
        "tag": "k100_compare_top1500",
        "learned_npz": BASE_DIR / "learned_lif_outputs" / "connectivity_20260423_091632_k100_l10_md8_vf20_e40_hybrid_sf80_lag8.npz",
        "learned_pt": BASE_DIR / "learned_lif_outputs" / "learned_lif_20260423_091632_k100_l10_md8_vf20_e40_hybrid_sf80_lag8.pt",
    },
    {
        "label": "0425 dataset",
        "session": "20260425_110211",
        "tag": "k100_compare_top1500",
        "learned_npz": BASE_DIR / "learned_lif_outputs" / "connectivity_20260425_110211_k100_l10_md8_vf20_e40_hybrid_sf80_lag8.npz",
        "learned_pt": BASE_DIR / "learned_lif_outputs" / "learned_lif_20260425_110211_k100_l10_md8_vf20_e40_hybrid_sf80_lag8.pt",
    },
    {
        "label": "0426 dataset",
        "session": "20260426_190829",
        "tag": "k100_compare_top1500",
        "learned_npz": BASE_DIR / "learned_lif_outputs" / "connectivity_20260426_190829_k100_l10_md8_vf20_e40_hybrid_sf80_lag8.npz",
        "learned_pt": BASE_DIR / "learned_lif_outputs" / "learned_lif_20260426_190829_k100_l10_md8_vf20_e40_hybrid_sf80_lag8.pt",
    },
]


def dataset_figure_paths(dataset):
    output_name = f'{dataset["session"]}_{dataset["tag"]}'
    corr_dir = BASE_DIR / "correlation_outputs"
    return {
        "true": corr_dir / f'true_connectivity_map_{output_name}.png',
        "lagged": corr_dir / f'lagged_correlation_map_{output_name}.png',
        "learned": corr_dir / f'learned_lif_connectivity_map_{output_name}.png',
        "correlation_npz": corr_dir / f'correlation_connectivity_{output_name}.npz',
    }


def format_metric(value, digits=4):
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(number) or np.isinf(number):
        return "n/a"
    return f"{number:.{digits}f}"


def load_learned_parameter_summary(dataset):
    ckpt = torch.load(str(dataset["learned_pt"]), map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    corr_data = np.load(dataset_figure_paths(dataset)["correlation_npz"], allow_pickle=True)

    alpha_val = float(torch.sigmoid(state["alpha_logit"]).item())
    tau_eff = float(-1.0 / np.log(alpha_val + 1e-10))
    threshold = float(state["threshold"].item())
    beta = float(state["beta"].item())
    reset = float(torch.nn.functional.softplus(state["reset_strength"]).item())
    candidate_info = ckpt.get("candidate_info", {})

    learned_edges = "n/a"
    corr_path = corr_data["learned_lif_path"].item() if getattr(corr_data["learned_lif_path"], "shape", ()) == () else corr_data["learned_lif_path"]
    if corr_path:
        learned_corr = np.load(corr_path, allow_pickle=True)
        conn = np.abs(learned_corr["connectivity_matrix"])
        learned_thresh = float(learned_corr["threshold"])
        learned_edges = str(int(np.count_nonzero(conn >= learned_thresh)))
    else:
        learned_thresh = np.nan

    summary = {
        "alpha": alpha_val,
        "tau_ms": tau_eff,
        "threshold": threshold,
        "beta": beta,
        "reset": reset,
        "learned_threshold": learned_thresh,
        "learned_auc": float(corr_data["learned_lif_eval_auc"]),
        "learned_ap": float(corr_data["learned_lif_eval_ap"]),
        "lagged_auc": float(corr_data["lagged_eval_auc"]),
        "lagged_ap": float(corr_data["lagged_eval_ap"]),
        "top_k": int(corr_data["top_k"]),
        "edge_cutoff": float(corr_data["score_cutoff"]),
        "bin_ms": float(corr_data["bin_size_ms"]),
        "lag_window_ms": float(corr_data["max_lag_ms"]),
        "source_note": corr_data["source_note"].item() if getattr(corr_data["source_note"], "shape", ()) == () else str(corr_data["source_note"]),
        "candidate_mode": candidate_info.get("mode", "hybrid"),
        "spatial_k": candidate_info.get("n_spatial", "n/a"),
        "temporal_k": candidate_info.get("n_temporal", "n/a"),
        "max_delay": ckpt.get("max_delay", "n/a"),
        "k": ckpt.get("K", "n/a"),
        "learned_edges": learned_edges,
    }
    return summary


def build_dataset_takeaway_text(summary):
    return (
        f'Learned-LIF lifts AP {format_metric(summary["lagged_ap"], 3)} -> '
        f'{format_metric(summary["learned_ap"], 3)} and AUC '
        f'{format_metric(summary["lagged_auc"], 3)} -> {format_metric(summary["learned_auc"], 3)}; '
        'the lagged map is a coarse baseline rather than the final connectivity estimate.'
    )


def set_speaker_notes(slide, lines):
    notes_frame = slide.notes_slide.notes_text_frame
    notes_frame.clear()
    for idx, line in enumerate(lines):
        paragraph = notes_frame.paragraphs[0] if idx == 0 else notes_frame.add_paragraph()
        paragraph.text = line


def build_truth_slide_notes(dataset, summary):
    return [
        f'{dataset["label"]}: baseline connectivity comparison.',
        'The left map is the full ground-truth undirected edge set saved from the simulation.',
        f'The right map is the lagged-correlation baseline restricted to the top {summary["top_k"]} undirected edges.',
        f'Lagged correlation uses {summary["bin_ms"]:.0f} ms bins with a {summary["lag_window_ms"]:.0f} ms positive lag window and reaches AUC {summary["lagged_auc"]:.3f}, AP {summary["lagged_ap"]:.3f}.',
        'Use this slide to show what the simple burst-synchrony baseline captures before moving to the learned-LIF estimate on the next slide.',
    ]


def build_learned_slide_notes(dataset, summary):
    return [
        f'{dataset["label"]}: learned-LIF K={summary["k"]} with a hybrid candidate set.',
        f'The candidate pool is {summary["spatial_k"]} spatial plus {summary["temporal_k"]} temporal neighbors, with max delay {summary["max_delay"]} bins.',
        f'Learned membrane parameters are alpha {summary["alpha"]:.4f}, tau_m about {summary["tau_ms"]:.1f} ms, threshold {summary["threshold"]:.4f}, beta {summary["beta"]:.4f}, and reset {summary["reset"]:.4f}.',
        f'The thresholded learned map reaches AUC {summary["learned_auc"]:.3f} and AP {summary["learned_ap"]:.3f}, above the lagged baseline for this session.',
        build_dataset_takeaway_text(summary),
    ]


def build_simulation_slide_notes():
    return [
        'Simulation slide: establish that the inference problem is grounded in known network structure rather than unknown biology.',
        'The network is clustered, distance-dependent, and driven by stimulation so we get burst structure plus ground-truth connectivity labels.',
        'The important point is that we save both the exact network graph and the raw spike recordings, so later recovery metrics have a trusted target.',
        'Use the raster on the right as the visual anchor: the same session can be viewed at acquisition-like resolution for presentation and at full spike-time resolution for modeling.',
    ]


def build_preprocess_slide_notes():
    return [
        'Preprocessing slide: explain why we do not train on full sparse recordings directly.',
        'First, all recordings from the same network are concatenated into one 1 ms binary spike matrix while preserving recording boundaries.',
        'Then we propose a hybrid candidate set and cut positive windows around true postsynaptic spikes plus negative windows from spike-free times.',
        'The warmup segment lets the membrane state settle, and the event-centered window forces optimization to focus on informative causal context instead of all-zero background.',
    ]


def build_pipeline_slide_notes():
    return [
        'Pipeline slide: walk left to right rather than reading every box in detail.',
        'The main spike-only path is recordings to binning to candidate proposal to event-window datasets to per-neuron fitting.',
        'The voltage branch is optional and only changes the supervision signal; it does not replace the presynaptic spike features.',
        'This slide is the high-level function map that connects the later detailed equations and results back to code structure.',
    ]


def build_training_slide_notes():
    return [
        'Training strategy slide: emphasize why the split is by recordings when possible, not by neurons.',
        'Each postsynaptic neuron owns one learned weight row, so holding out neurons would test rows that were never trained.',
        'Positive and negative event windows create a balanced supervised problem around actual spike decisions, which avoids the trivial all-silence solution.',
        'The bottom ribbon is the temporal layout the model sees during each sample: warmup, causal pre-spike context, then a short post-spike reset segment.',
    ]


def build_model_slide_notes():
    return [
        'Model slide: this is the differentiable LIF core used for inference rather than forward simulation of the original network.',
        'Delayed presynaptic inputs are mixed with learned delay probabilities, summed through learned weights, integrated by alpha, then passed through a soft threshold.',
        'Connectivity is read out from the learned absolute weight magnitude after training, while the global membrane parameters stay shared across neurons in this version.',
        'The right side of the slide ties the equations back to implementation details and the current operating setup used for the reported results.',
    ]


def build_voltage_slide_notes():
    return [
        'Voltage slide: clarify that voltage augmentation is extra supervision, not a different presynaptic representation.',
        'We still use spike trains as input, but we clean the postsynaptic voltage trace and add a subthreshold reconstruction term to the loss.',
        'This gives the optimizer access to membrane dynamics between spikes, which improves some recovery metrics and sign information.',
        'The key comparison to state aloud is that voltage helps AUROC and recall, while the spike-only model can still be competitive on AP depending on the session.',
    ]


def build_results_slide_notes():
    return [
        'Results slide: summarize the numeric story before showing the qualitative maps.',
        'The spike-only learned-LIF runs on 0423 and 0425 already recover directed connectivity substantially above simple baselines.',
        'The voltage-augmented run on 0425 pushes AUROC higher and improves sign and weight-scale recovery, even though not every metric moves in the same direction.',
        'Use this slide to set up the next section, where the audience can inspect what these metrics look like as actual network maps.',
    ]


def build_tradeoff_slide_notes():
    return [
        'Tradeoff slide: this is where to explain why we did not simply keep increasing K.',
        'The qualitative map on the left shows that the model can recover both local within-cluster structure and some longer-range edges.',
        'The tradeoff panel shows that larger candidate sets increase coverage but can hurt precision-recall behavior once too many distractor edges are introduced.',
        'That is why the project currently treats the smaller hybrid candidate set as the practical operating point, even though K=100 is shown for comparison in the later slides.',
    ]


def rgb(hex_color):
    hex_color = hex_color.replace("#", "")
    return RGBColor(int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


PALETTE = {
    "ink": rgb("13202B"),
    "ink_2": rgb("1B2D3A"),
    "sand": rgb("F4F1EA"),
    "paper": rgb("FFFDF8"),
    "teal": rgb("1E7DAA"),
    "forest": rgb("2F7D4A"),
    "gold": rgb("E6B655"),
    "brick": rgb("C2653F"),
    "plum": rgb("2A2F6B"),
    "slate": rgb("5D6B78"),
    "muted": rgb("D8E0E8"),
    "dark_text": rgb("1E252B"),
    "light_text": rgb("F6F7F7"),
}

TITLE_FONT = "Georgia"
BODY_FONT = "Calibri"
MONO_FONT = "Consolas"


def add_full_background(slide, color):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE,
        0,
        0,
        SLIDE_WIDTH,
        SLIDE_HEIGHT,
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_textbox(
    slide,
    text,
    x,
    y,
    w,
    h,
    *,
    font_name=BODY_FONT,
    font_size=18,
    color=None,
    bold=False,
    italic=False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0.0,
    line_spacing=1.1,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = valign
    margin_pt = Pt(margin * 72)
    frame.margin_left = margin_pt
    frame.margin_right = margin_pt
    frame.margin_top = margin_pt
    frame.margin_bottom = margin_pt

    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = line_spacing
    run = paragraph.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color or PALETTE["dark_text"]
    return box


def add_bullets(
    slide,
    items,
    x,
    y,
    w,
    h,
    *,
    font_size=16,
    color=None,
    level=0,
    space_after=5,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Pt(2)
    frame.margin_right = Pt(2)
    frame.margin_top = Pt(2)
    frame.margin_bottom = Pt(2)
    color = color or PALETTE["dark_text"]

    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = item
        paragraph.level = level
        paragraph.bullet = True
        paragraph.space_after = Pt(space_after)
        paragraph.line_spacing = 1.15
        paragraph.font.name = BODY_FONT
        paragraph.font.size = Pt(font_size)
        paragraph.font.color.rgb = color
    return box


def add_panel(slide, x, y, w, h, fill, *, line=None, radius=True):
    shape_type = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    panel = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill
    panel.line.color.rgb = line or fill
    panel.line.width = Pt(1.2)
    return panel


def add_stat_card(slide, x, y, w, h, fill, label, value, *, value_size=24):
    add_panel(slide, x, y, w, h, fill)
    add_textbox(
        slide,
        label,
        x + 0.12,
        y + 0.10,
        w - 0.24,
        0.24,
        font_name=BODY_FONT,
        font_size=10,
        color=PALETTE["light_text"],
        margin=0,
    )
    add_textbox(
        slide,
        value,
        x + 0.14,
        y + 0.38,
        w - 0.28,
        h - 0.46,
        font_name=BODY_FONT,
        font_size=value_size,
        color=PALETTE["light_text"],
        bold=True,
        margin=0,
    )


def add_tag(slide, text, x, y, w, h, fill, text_color):
    add_panel(slide, x, y, w, h, fill)
    add_textbox(
        slide,
        text,
        x + 0.1,
        y + 0.02,
        w - 0.2,
        h - 0.04,
        font_name=BODY_FONT,
        font_size=11,
        color=text_color,
        bold=True,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        margin=0,
    )


def add_image_contain(slide, path, x, y, w, h, *, border_color=None, bg_color=None):
    if bg_color is not None:
        frame = add_panel(slide, x, y, w, h, bg_color, line=border_color or bg_color, radius=False)
        frame.line.width = Pt(1.0)

    with Image.open(path) as image:
        image_aspect = image.width / image.height

    box_aspect = w / h
    if image_aspect >= box_aspect:
        draw_w = w
        draw_h = w / image_aspect
        draw_x = x
        draw_y = y + (h - draw_h) / 2.0
    else:
        draw_h = h
        draw_w = h * image_aspect
        draw_x = x + (w - draw_w) / 2.0
        draw_y = y

    slide.shapes.add_picture(str(path), Inches(draw_x), Inches(draw_y), width=Inches(draw_w), height=Inches(draw_h))


def add_network_motif(slide, x, y, scale=1.0):
    points = [
        (x + 0.20 * scale, y + 0.15 * scale),
        (x + 0.95 * scale, y + 0.50 * scale),
        (x + 0.55 * scale, y + 1.05 * scale),
        (x + 1.30 * scale, y + 0.95 * scale),
        (x + 1.55 * scale, y + 0.35 * scale),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (1, 4)]
    for start_idx, end_idx in edges:
        start = points[start_idx]
        end = points[end_idx]
        connector = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            Inches(start[0]),
            Inches(start[1]),
            Inches(end[0]),
            Inches(end[1]),
        )
        connector.line.color.rgb = PALETTE["teal"]
        connector.line.width = Pt(2.2)

    for idx, (px, py) in enumerate(points):
        node = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(px - 0.07), Inches(py - 0.07), Inches(0.14), Inches(0.14))
        node.fill.solid()
        node.fill.fore_color.rgb = PALETTE["gold"] if idx in {1, 4} else rgb("77C4C9")
        node.line.fill.background()


def add_footer(slide, text, slide_num, *, dark=False):
    color = PALETTE["muted"] if dark else PALETTE["slate"]
    add_textbox(slide, text, 0.6, 7.0, 3.0, 0.22, font_name=BODY_FONT, font_size=9, color=color, italic=True)
    add_textbox(slide, f"{slide_num}", 12.2, 6.95, 0.5, 0.22, font_name=BODY_FONT, font_size=10, color=color, align=PP_ALIGN.RIGHT)


def build_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["ink"])
    add_textbox(
        slide,
        "Clustered LIF Simulation and\nLearned-LIF Connectivity Inference",
        0.7,
        0.65,
        6.4,
        1.55,
        font_name=TITLE_FONT,
        font_size=28,
        color=PALETTE["light_text"],
        bold=True,
        margin=0,
        line_spacing=1.0,
    )
    add_textbox(
        slide,
        "Simulation design, data preprocessing, training strategy, model structure, and current results",
        0.75,
        2.20,
        7.0,
        0.45,
        font_name=BODY_FONT,
        font_size=17,
        color=PALETTE["muted"],
        margin=0,
    )
    add_stat_card(slide, 0.75, 3.05, 1.90, 0.90, PALETTE["plum"], "Simulation", "dt = 0.1 ms", value_size=18)
    add_stat_card(slide, 2.85, 3.05, 1.90, 0.90, PALETTE["teal"], "Inference", "hybrid K = 50", value_size=17)
    add_stat_card(slide, 4.95, 3.05, 1.95, 0.90, PALETTE["forest"], "Best recent", "AUROC = 0.885", value_size=17)
    add_textbox(
        slide,
        "Goal: generate stimulus-driven ground-truth networks and recover directed connectivity with an interpretable differentiable LIF model, optionally supervised by cleaned subthreshold voltage.",
        0.75,
        4.45,
        6.7,
        1.0,
        font_name=BODY_FONT,
        font_size=17,
        color=PALETTE["light_text"],
        margin=0,
        line_spacing=1.2,
    )
    add_network_motif(slide, 9.0, 0.95, scale=1.85)
    add_footer(slide, "April 2026", 1, dark=True)


def build_simulation_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["sand"])
    add_tag(slide, "SIMULATION ENGINE", 0.65, 0.45, 1.65, 0.32, PALETTE["teal"], PALETTE["light_text"])
    add_textbox(slide, "Clustered network design and saved outputs", 0.65, 0.88, 5.8, 0.6, font_name=TITLE_FONT, font_size=24, color=PALETTE["dark_text"], bold=True, margin=0)

    add_panel(slide, 0.70, 1.65, 3.25, 1.05, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Clustered topology", 0.88, 1.82, 1.8, 0.25, font_name=BODY_FONT, font_size=12, color=PALETTE["teal"], bold=True, margin=0)
    add_textbox(slide, "20 clusters, distance-dependent connectivity, excitatory/inhibitory weights, and stimulus-driven bursts.", 0.88, 2.05, 2.8, 0.50, font_name=BODY_FONT, font_size=15, color=PALETTE["dark_text"], margin=0)

    add_panel(slide, 0.70, 2.95, 3.25, 1.05, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Temporal resolution", 0.88, 3.12, 1.9, 0.25, font_name=BODY_FONT, font_size=12, color=PALETTE["forest"], bold=True, margin=0)
    add_textbox(slide, "Simulate at dt = 0.1 ms, then save raw spike times and acquisition-style resampled activity for comparison plots.", 0.88, 3.35, 2.8, 0.50, font_name=BODY_FONT, font_size=15, color=PALETTE["dark_text"], margin=0)

    add_panel(slide, 0.70, 4.25, 3.25, 1.25, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Saved per session", 0.88, 4.42, 1.8, 0.25, font_name=BODY_FONT, font_size=12, color=PALETTE["brick"], bold=True, margin=0)
    add_bullets(
        slide,
        [
            "network_TIMESTAMP.npz: true connectivity and neuron positions",
            "recordingXXX.npz: spikes, optional raw voltage, metadata",
            "multiple recordings per network support held-out recording validation",
        ],
        0.88,
        4.65,
        2.85,
        0.72,
        font_size=14,
        color=PALETTE["dark_text"],
        space_after=2,
    )

    add_image_contain(
        slide,
        ASSETS["bursts"],
        4.35,
        1.50,
        8.15,
        4.95,
        border_color=rgb("B7C3CE"),
        bg_color=PALETTE["paper"],
    )
    add_textbox(
        slide,
        "Example burst raster after coarse resampling. The same sessions also retain the full-resolution spike trains used by the inference model.",
        4.40,
        6.55,
        7.95,
        0.42,
        font_name=BODY_FONT,
        font_size=13,
        color=PALETTE["slate"],
        margin=0,
    )
    set_speaker_notes(slide, build_simulation_slide_notes())
    add_footer(slide, "Simulation to saved sessions", 2)


def build_preprocess_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["paper"])
    add_tag(slide, "DATA PREPROCESS", 0.65, 0.45, 1.50, 0.32, PALETTE["forest"], PALETTE["light_text"])
    add_textbox(slide, "How spikes become training windows", 0.65, 0.88, 6.3, 0.54, font_name=TITLE_FONT, font_size=22, color=PALETTE["dark_text"], bold=True, margin=0)

    add_panel(slide, 0.70, 1.55, 3.50, 2.20, PALETTE["sand"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "1. Session to spike matrix", 0.95, 1.78, 2.4, 0.24, font_name=BODY_FONT, font_size=14, color=PALETTE["teal"], bold=True, margin=0)
    add_bullets(
        slide,
        [
            "load_all_recordings() joins recordings from one session.",
            "spike_times_to_binary() bins spikes at 1 ms.",
            "Boundaries prevent cross-recording windows.",
        ],
        0.92,
        2.08,
        3.0,
        1.2,
        font_size=12,
        color=PALETTE["dark_text"],
        space_after=2,
    )
    add_textbox(slide, "dt = 1 ms, K = 50, max_delay = 8", 0.95, 3.50, 2.65, 0.14, font_name=MONO_FONT, font_size=9, color=PALETTE["slate"], margin=0)

    add_panel(slide, 0.70, 3.95, 3.50, 2.35, PALETTE["sand"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "2. Candidate proposal", 0.95, 4.18, 2.4, 0.24, font_name=BODY_FONT, font_size=14, color=PALETTE["forest"], bold=True, margin=0)
    add_bullets(
        slide,
        [
            "compute_neighbor_indices() builds a hybrid candidate set for each postsynaptic neuron.",
            "By default: 40 spatial neighbors + 10 temporal candidates.",
            "Temporal candidates come from spikes that lead the postsynaptic spike by 1 to 8 bins.",
        ],
        0.92,
        4.48,
        3.0,
        1.30,
        font_size=12,
        color=PALETTE["dark_text"],
        space_after=2,
    )

    add_panel(slide, 4.50, 1.55, 8.20, 4.50, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "3. Positive and negative event windows", 4.78, 1.78, 3.5, 0.24, font_name=BODY_FONT, font_size=14, color=PALETTE["brick"], bold=True, margin=0)
    add_textbox(slide, "find_event_windows() builds a short window around a valid center time t.", 4.78, 2.08, 5.6, 0.20, font_name=BODY_FONT, font_size=12, color=PALETTE["dark_text"], margin=0)
    add_textbox(slide, "L = warmup + pre_context + post_context = 90 bins", 4.78, 2.30, 5.6, 0.18, font_name=MONO_FONT, font_size=13, color=PALETTE["plum"], margin=0)

    add_panel(slide, 4.85, 2.82, 3.45, 1.45, PALETTE["sand"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Positive window", 5.10, 2.96, 1.5, 0.22, font_name=BODY_FONT, font_size=13, color=PALETTE["teal"], bold=True, margin=0)
    add_textbox(slide, "For each real postsynaptic spike at bin t, keep the window only if it fits inside one recording.", 5.10, 3.20, 2.8, 0.34, font_name=BODY_FONT, font_size=10, color=PALETTE["dark_text"], margin=0, line_spacing=1.0)
    add_textbox(slide, "window = [t - 80, t + 10)", 5.10, 3.68, 2.8, 0.16, font_name=MONO_FONT, font_size=11, color=PALETTE["plum"], margin=0)

    add_panel(slide, 8.55, 2.82, 3.45, 1.45, PALETTE["sand"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Negative window", 8.80, 2.96, 1.5, 0.22, font_name=BODY_FONT, font_size=13, color=PALETTE["forest"], bold=True, margin=0)
    add_textbox(slide, "Sample a center t from a valid range, then keep it only if the postsynaptic neuron has no spike within ± neg_min_distance.", 8.80, 3.20, 2.8, 0.40, font_name=BODY_FONT, font_size=10, color=PALETTE["dark_text"], margin=0, line_spacing=1.0)
    add_textbox(slide, "neg_min_distance = 100", 8.80, 3.68, 2.6, 0.16, font_name=MONO_FONT, font_size=11, color=PALETTE["plum"], margin=0)

    add_textbox(slide, "warmup", 5.00, 4.46, 0.85, 0.14, font_name=BODY_FONT, font_size=10, color=PALETTE["plum"], bold=True, align=PP_ALIGN.CENTER, margin=0)
    add_textbox(slide, "pre", 6.18, 4.46, 1.30, 0.14, font_name=BODY_FONT, font_size=10, color=PALETTE["teal"], bold=True, align=PP_ALIGN.CENTER, margin=0)
    add_textbox(slide, "post", 7.85, 4.46, 0.85, 0.14, font_name=BODY_FONT, font_size=10, color=PALETTE["forest"], bold=True, align=PP_ALIGN.CENTER, margin=0)
    add_panel(slide, 5.00, 4.70, 1.00, 0.24, PALETTE["plum"], line=PALETTE["plum"], radius=False)
    add_panel(slide, 6.05, 4.70, 1.80, 0.24, PALETTE["teal"], line=PALETTE["teal"], radius=False)
    add_panel(slide, 7.90, 4.70, 0.80, 0.24, PALETTE["forest"], line=PALETTE["forest"], radius=False)
    add_textbox(slide, "event spike here", 6.12, 4.98, 2.5, 0.16, font_name=BODY_FONT, font_size=11, color=PALETTE["brick"], italic=True, margin=0)

    add_panel(slide, 4.85, 5.30, 7.15, 0.80, PALETTE["ink_2"], line=PALETTE["ink_2"], radius=False)
    add_textbox(slide, "Why windows? Full recordings are dominated by zeros. Event windows force the optimizer to spend time on informative causal contexts, while warmup bins let the membrane settle before the loss is computed.", 5.05, 5.50, 6.7, 0.32, font_name=BODY_FONT, font_size=11, color=PALETTE["light_text"], margin=0, line_spacing=1.0)

    set_speaker_notes(slide, build_preprocess_slide_notes())
    add_footer(slide, "Event-window construction", 3)


def build_pipeline_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["sand"])
    add_tag(slide, "PIPELINE", 0.65, 0.45, 1.05, 0.32, PALETTE["teal"], PALETTE["light_text"])
    add_textbox(slide, "Pipeline architecture", 0.65, 0.88, 4.8, 0.54, font_name=TITLE_FONT, font_size=22, color=PALETTE["dark_text"], bold=True, margin=0)

    boxes = [
        (0.80, 1.95, 2.00, 1.40, PALETTE["paper"], "Session files", "network_*.npz\nrecordingXXX.npz", "load recordings"),
        (3.15, 1.95, 2.00, 1.40, PALETTE["paper"], "Spike preprocess", "1 ms binary matrix\nrecording boundaries kept", "bin spikes"),
        (5.50, 1.95, 2.00, 1.40, PALETTE["paper"], "Candidates + windows", "hybrid K neighbors\npos/neg event windows", "neighbors + windows"),
        (7.85, 1.95, 2.00, 1.40, PALETTE["paper"], "Datasets + split", "same-neuron training\nheld-out recordings if possible", "train/val split"),
        (10.20, 1.95, 2.10, 1.40, PALETTE["paper"], "Model + loss", "soft LIF forward pass\nBCE + L1", "fit per neuron"),
    ]
    for x, y, w, h, fill, title, body, fn in boxes:
        add_panel(slide, x, y, w, h, fill, line=rgb("D8D1C5"), radius=False)
        add_textbox(slide, title, x + 0.12, y + 0.12, w - 0.24, 0.22, font_name=BODY_FONT, font_size=13, color=PALETTE["teal"], bold=True, align=PP_ALIGN.CENTER, margin=0)
        add_textbox(slide, body, x + 0.12, y + 0.42, w - 0.24, 0.34, font_name=BODY_FONT, font_size=12, color=PALETTE["dark_text"], align=PP_ALIGN.CENTER, margin=0, line_spacing=1.0)
    for x in [2.85, 5.20, 7.55, 9.90]:
        add_textbox(slide, "→", x, 2.42, 0.20, 0.18, font_name=BODY_FONT, font_size=22, color=PALETTE["slate"], bold=True, align=PP_ALIGN.CENTER, margin=0)

    add_panel(slide, 2.35, 4.15, 2.35, 1.35, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Optional voltage branch", 2.52, 4.34, 2.0, 0.22, font_name=BODY_FONT, font_size=13, color=PALETTE["brick"], bold=True, align=PP_ALIGN.CENTER, margin=0)
    add_textbox(slide, "clean raw voltage\nmask spike neighborhoods\nnormalize per neuron", 2.52, 4.63, 2.0, 0.34, font_name=BODY_FONT, font_size=12, color=PALETTE["dark_text"], align=PP_ALIGN.CENTER, margin=0, line_spacing=1.0)

    add_textbox(slide, "↓", 3.38, 3.55, 0.20, 0.20, font_name=BODY_FONT, font_size=18, color=PALETTE["slate"], bold=True, align=PP_ALIGN.CENTER, margin=0)
    add_textbox(slide, "↗", 5.00, 4.55, 0.20, 0.20, font_name=BODY_FONT, font_size=18, color=PALETTE["slate"], bold=True, align=PP_ALIGN.CENTER, margin=0)

    add_panel(slide, 6.05, 4.15, 3.05, 1.35, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Voltage-augmented model", 6.25, 4.34, 2.65, 0.22, font_name=BODY_FONT, font_size=13, color=PALETTE["forest"], bold=True, align=PP_ALIGN.CENTER, margin=0)
    add_textbox(slide, "same spike inputs\nplus SmoothL1 voltage supervision", 6.25, 4.63, 2.65, 0.30, font_name=BODY_FONT, font_size=12, color=PALETTE["dark_text"], align=PP_ALIGN.CENTER, margin=0, line_spacing=1.0)

    add_panel(slide, 9.45, 4.15, 2.85, 1.35, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Outputs", 9.65, 4.34, 2.45, 0.22, font_name=BODY_FONT, font_size=13, color=PALETTE["plum"], bold=True, align=PP_ALIGN.CENTER, margin=0)
    add_textbox(slide, "|W| scores\nAUROC, AP, F1\nfigures and .npz artifacts", 9.65, 4.63, 2.45, 0.38, font_name=BODY_FONT, font_size=12, color=PALETTE["dark_text"], align=PP_ALIGN.CENTER, margin=0, line_spacing=1.0)

    set_speaker_notes(slide, build_pipeline_slide_notes())
    add_footer(slide, "Pipeline and function map", 4)


def build_training_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["sand"])
    add_tag(slide, "TRAINING STRATEGY", 0.65, 0.45, 1.70, 0.32, PALETTE["brick"], PALETTE["light_text"])
    add_textbox(slide, "Why the model trains on short windows instead of full sequences", 0.65, 0.88, 7.5, 0.6, font_name=TITLE_FONT, font_size=24, color=PALETTE["dark_text"], bold=True, margin=0)

    cards = [
        (0.75, 1.75, 2.90, 1.95, PALETTE["paper"], "1. Candidate proposal", [
            "Hybrid K=50 is the current operating point.",
            "Mostly nearby neurons plus a temporal shortlist.",
            "Recent coverage on held-out sessions: about 86-89%.",
        ], PALETTE["teal"]),
        (3.95, 1.75, 2.90, 1.95, PALETTE["paper"], "2. Event-window sampling", [
            "Positive windows center on real postsynaptic spikes.",
            "Negative windows come from spike-free times.",
            "Warmup bins let membrane state settle before loss.",
        ], PALETTE["forest"]),
        (7.15, 1.75, 2.90, 1.95, PALETTE["paper"], "3. Validation split", [
            "Prefer held-out recordings when multiple recordings exist.",
            "Recent runs: 16 training / 4 validation recordings.",
            "Otherwise split windows within the same neuron.",
        ], PALETTE["brick"]),
        (10.35, 1.75, 2.20, 1.95, PALETTE["paper"], "4. Why not hold out neurons?", [
            "Each postsynaptic neuron owns one weight row.",
            "Holding out neurons would score untrained rows.",
        ], PALETTE["plum"]),
    ]
    for x, y, w, h, fill, title, bullets, accent in cards:
        add_panel(slide, x, y, w, h, fill, line=rgb("D8D1C5"), radius=False)
        add_textbox(slide, title, x + 0.18, y + 0.16, w - 0.36, 0.28, font_name=BODY_FONT, font_size=13, color=accent, bold=True, margin=0)
        add_bullets(slide, bullets, x + 0.16, y + 0.48, w - 0.32, h - 0.58, font_size=13, color=PALETTE["dark_text"], space_after=1)

    add_panel(slide, 1.05, 4.05, 11.30, 1.55, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Window layout used during training", 1.30, 4.22, 2.3, 0.25, font_name=BODY_FONT, font_size=13, color=PALETTE["slate"], bold=True, margin=0)
    segments = [
        ("warmup", 3.25, 1.45, PALETTE["plum"]),
        ("pre-spike context", 4.80, 2.60, PALETTE["teal"]),
        ("post-spike / reset", 7.55, 2.25, PALETTE["forest"]),
    ]
    for label, x, w, fill in segments:
        add_panel(slide, x, 4.55, w, 0.42, fill, line=fill, radius=False)
        add_textbox(slide, label, x, 4.65, w, 0.14, font_name=BODY_FONT, font_size=12, color=PALETTE["light_text"], bold=True, align=PP_ALIGN.CENTER, margin=0)
    add_textbox(
        slide,
        "This avoids the trivial all-silence solution that dominates full-recording training on sparse spike trains.",
        1.30,
        5.18,
        10.4,
        0.26,
        font_name=BODY_FONT,
        font_size=14,
        color=PALETTE["dark_text"],
        margin=0,
    )
    set_speaker_notes(slide, build_training_slide_notes())
    add_footer(slide, "Event-window training", 5)


def build_model_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["ink_2"])
    add_tag(slide, "MODEL", 0.65, 0.45, 1.00, 0.32, PALETTE["gold"], PALETTE["dark_text"])
    add_textbox(slide, "Model equations, parameters, and main functions", 0.65, 0.88, 7.7, 0.56, font_name=TITLE_FONT, font_size=23, color=PALETTE["light_text"], bold=True, margin=0)

    add_panel(slide, 0.78, 1.62, 7.25, 5.10, PALETTE["paper"], line=rgb("A9B8C2"), radius=False)
    add_textbox(slide, "Core equations", 1.02, 1.86, 1.8, 0.22, font_name=BODY_FONT, font_size=14, color=PALETTE["teal"], bold=True, margin=0)
    equations = [
        "x_delay,j,i(t) = Σ_d p_j,i(d) x_i(t-d)",
        "I_j(t) = Σ_i W_j,i x_delay,j,i(t)",
        "V_j(t) = α V_j(t-1) + I_j(t)",
        "ŝ_j(t) = σ(β (V_j(t) - θ))",
        "V_j(t) ← V_j(t) - r ŝ_j(t)",
        "Spike-only: L = BCE(ŝ, s) + λ_1 ||W||_1",
        "Voltage-augmented: L = BCE(ŝ, s) + λ_v SmoothL1(V, V*) + λ_1 ||W||_1",
    ]
    y = 2.18
    for eq in equations:
        add_textbox(slide, eq, 1.05, y, 6.45, 0.28, font_name=MONO_FONT, font_size=16, color=PALETTE["plum"], margin=0)
        y += 0.50

    add_panel(slide, 8.28, 1.62, 4.25, 2.10, PALETTE["paper"], line=rgb("A9B8C2"), radius=False)
    add_textbox(slide, "Parameters", 8.52, 1.86, 1.4, 0.22, font_name=BODY_FONT, font_size=14, color=PALETTE["forest"], bold=True, margin=0)
    add_bullets(
        slide,
        [
            "Per neuron: W[j,i] and delay_logits[j,i,d] for candidate edges.",
            "Shared globals: α, θ, β, and reset strength r.",
            "Voltage model adds a per-neuron bias and a voltage loss term.",
        ],
        8.45,
        2.16,
        3.55,
        1.20,
        font_size=13,
        color=PALETTE["dark_text"],
        space_after=2,
    )

    add_panel(slide, 8.28, 3.98, 4.25, 1.55, PALETTE["paper"], line=rgb("A9B8C2"), radius=False)
    add_textbox(slide, "Main functions", 8.52, 4.20, 1.8, 0.22, font_name=BODY_FONT, font_size=14, color=PALETTE["brick"], bold=True, margin=0)
    add_textbox(slide, "PerNeuronLIF.forward()\ntrain_epoch_events()\nevaluate_event_windows()\nevaluate_connectivity()", 8.52, 4.52, 3.35, 0.82, font_name=MONO_FONT, font_size=11, color=PALETTE["slate"], margin=0, line_spacing=1.0)

    add_panel(slide, 8.28, 5.70, 4.25, 0.88, PALETTE["paper"], line=rgb("A9B8C2"), radius=False)
    add_textbox(slide, "Current setup", 8.52, 5.88, 1.6, 0.20, font_name=BODY_FONT, font_size=14, color=PALETTE["teal"], bold=True, margin=0)
    add_textbox(slide, "K = 50, D = 8, warmup/pre/post = 30/50/10, pos_weight = 5, λ_1 = 0.01", 8.52, 6.14, 3.55, 0.18, font_name=MONO_FONT, font_size=10, color=PALETTE["slate"], margin=0)
    set_speaker_notes(slide, build_model_slide_notes())
    add_footer(slide, "Equations and parameters", 6, dark=True)


def build_voltage_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["paper"])
    add_tag(slide, "VOLTAGE AUGMENTATION", 0.65, 0.45, 2.05, 0.32, PALETTE["plum"], PALETTE["light_text"])
    add_textbox(slide, "Voltage supervision adds subthreshold information without changing the presynaptic inputs", 0.65, 0.88, 11.2, 0.72, font_name=TITLE_FONT, font_size=20, color=PALETTE["dark_text"], bold=True, margin=0)

    add_image_contain(slide, ASSETS["voltage_traces"], 0.80, 1.65, 6.55, 4.95, border_color=rgb("B7C3CE"), bg_color=PALETTE["sand"])

    add_panel(slide, 7.65, 1.65, 4.90, 2.35, PALETTE["sand"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Key idea", 7.92, 1.87, 1.5, 0.26, font_name=BODY_FONT, font_size=14, color=PALETTE["teal"], bold=True, margin=0)
    add_bullets(
        slide,
        [
            "Presynaptic features remain spike trains.",
            "Postsynaptic voltage becomes an extra target after cleaning.",
            "The model is rewarded for matching both spike timing and subthreshold shape.",
        ],
        7.90,
        2.18,
        4.25,
        1.45,
        font_size=15,
        color=PALETTE["dark_text"],
        space_after=3,
    )

    add_panel(slide, 7.65, 4.20, 4.90, 1.20, PALETTE["sand"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Loss", 7.92, 4.40, 0.8, 0.20, font_name=BODY_FONT, font_size=14, color=PALETTE["brick"], bold=True, margin=0)
    add_textbox(slide, "L = BCE(spikes) + lambda_v MSE(voltage) + lambda_1 ||W||_1", 7.92, 4.72, 4.15, 0.36, font_name=MONO_FONT, font_size=15, color=PALETTE["dark_text"], margin=0)

    add_stat_card(slide, 7.70, 5.70, 1.45, 0.82, PALETTE["teal"], "AUROC", "0.885", value_size=20)
    add_stat_card(slide, 9.35, 5.70, 1.45, 0.82, PALETTE["forest"], "Sign acc.", "0.942", value_size=20)
    add_stat_card(slide, 11.00, 5.70, 1.45, 0.82, PALETTE["brick"], "Weight corr.", "0.773", value_size=20)

    add_textbox(
        slide,
        "Best current voltage-augmented run: session 20260425_110211, hybrid K=50, held-out recording validation.",
        7.80,
        6.68,
        4.55,
        0.24,
        font_name=BODY_FONT,
        font_size=13,
        color=PALETTE["slate"],
        margin=0,
    )
    set_speaker_notes(slide, build_voltage_slide_notes())
    add_footer(slide, "Voltage as supervision", 7)


def build_results_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["sand"])
    add_tag(slide, "RECENT RESULTS", 0.65, 0.45, 1.45, 0.32, PALETTE["forest"], PALETTE["light_text"])
    add_textbox(slide, "Held-out connectivity recovery on recent simulated sessions", 0.65, 0.88, 8.4, 0.54, font_name=TITLE_FONT, font_size=22, color=PALETTE["dark_text"], bold=True, margin=0)
    add_image_contain(slide, ASSETS["metrics"], 0.70, 1.55, 8.20, 5.20, border_color=rgb("B7C3CE"), bg_color=PALETTE["paper"])

    add_panel(slide, 9.15, 1.60, 3.45, 1.35, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Spike-only learned-LIF", 9.38, 1.82, 2.7, 0.24, font_name=BODY_FONT, font_size=14, color=PALETTE["teal"], bold=True, margin=0)
    add_textbox(slide, "0423: AUROC 0.863, AP 0.729, F1 0.686\n0425: AUROC 0.861, AP 0.743, F1 0.705", 9.38, 2.12, 2.9, 0.55, font_name=BODY_FONT, font_size=15, color=PALETTE["dark_text"], margin=0, line_spacing=1.15)

    add_panel(slide, 9.15, 3.15, 3.45, 1.50, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Voltage-augmented learned-LIF", 9.38, 3.37, 2.95, 0.24, font_name=BODY_FONT, font_size=14, color=PALETTE["forest"], bold=True, margin=0)
    add_textbox(slide, "0425: AUROC 0.885, AP 0.711, F1 0.710, Recall 0.698\nAlso recovered sign and weight scale more faithfully.", 9.38, 3.68, 2.9, 0.66, font_name=BODY_FONT, font_size=15, color=PALETTE["dark_text"], margin=0, line_spacing=1.15)

    add_panel(slide, 9.15, 4.85, 3.45, 1.35, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Main readout", 9.38, 5.07, 1.8, 0.22, font_name=BODY_FONT, font_size=14, color=PALETTE["brick"], bold=True, margin=0)
    add_textbox(slide, "Voltage supervision improves AUROC and recall, while the spike-only model still leads on AP for session 0425.", 9.38, 5.36, 2.88, 0.50, font_name=BODY_FONT, font_size=15, color=PALETTE["dark_text"], margin=0, line_spacing=1.15)
    set_speaker_notes(slide, build_results_slide_notes())
    add_footer(slide, "Held-out session metrics", 8)


def build_tradeoff_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["paper"])
    add_tag(slide, "QUALITATIVE VIEW", 0.65, 0.45, 1.55, 0.32, PALETTE["teal"], PALETTE["light_text"])
    add_textbox(slide, "Recovered connectivity structure and candidate-set tradeoff", 0.65, 0.88, 8.6, 0.54, font_name=TITLE_FONT, font_size=22, color=PALETTE["dark_text"], bold=True, margin=0)

    add_image_contain(slide, ASSETS["connectivity"], 0.70, 1.55, 8.35, 5.45, border_color=rgb("B7C3CE"), bg_color=PALETTE["sand"])
    add_image_contain(slide, ASSETS["tradeoff"], 9.25, 1.65, 3.15, 2.20, border_color=rgb("B7C3CE"), bg_color=PALETTE["sand"])

    add_panel(slide, 9.25, 4.00, 3.15, 2.30, PALETTE["sand"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Takeaways", 9.48, 4.32, 1.4, 0.22, font_name=BODY_FONT, font_size=14, color=PALETTE["brick"], bold=True, margin=0)
    add_bullets(
        slide,
        [
            "0423 shows many within-cluster and longer-range edges recovered after thresholding the learned weights.",
            "For the voltage model, K=50 beat K=100 on AP and F1 even though K=100 had higher coverage.",
            "The current best operating point is therefore the smaller hybrid candidate set.",
        ],
        9.42,
        4.62,
        2.75,
        1.48,
        font_size=12,
        color=PALETTE["dark_text"],
        space_after=2,
    )
    set_speaker_notes(slide, build_tradeoff_slide_notes())
    add_footer(slide, "K=50 precision-recall sweet spot", 15)


def build_close_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["ink"])
    add_textbox(slide, "Takeaways and next steps", 0.70, 0.70, 4.2, 0.7, font_name=TITLE_FONT, font_size=26, color=PALETTE["light_text"], bold=True, margin=0)
    add_textbox(slide, "The deck focuses on the pieces that matter most for discussion with collaborators: preprocessing, training strategy, model assumptions, and what the current metrics actually say.", 0.72, 1.55, 6.4, 0.7, font_name=BODY_FONT, font_size=16, color=PALETTE["muted"], margin=0, line_spacing=1.15)

    boxes = [
        (0.85, 2.55, 2.75, 1.55, PALETTE["plum"], "1", "Simulation", "Ground-truth clustered networks provide aligned spikes, voltage, and known connectivity."),
        (3.85, 2.55, 2.75, 1.55, PALETTE["teal"], "2", "Preprocess", "Spikes are binned and windowed; voltage is cleaned into a usable subthreshold target."),
        (6.85, 2.55, 2.75, 1.55, PALETTE["forest"], "3", "Inference", "Event-window learned-LIF recovers connectivity with interpretable parameters and competitive metrics."),
        (9.85, 2.55, 2.65, 1.55, PALETTE["brick"], "4", "Next", "Cross-session transfer, perturbation tests, and comparison with RiTINI-style forecasting remain open."),
    ]
    for x, y, w, h, fill, num, title, body in boxes:
        add_panel(slide, x, y, w, h, fill)
        add_textbox(slide, num, x + 0.16, y + 0.12, 0.35, 0.30, font_name=TITLE_FONT, font_size=22, color=PALETTE["light_text"], bold=True, margin=0)
        add_textbox(slide, title, x + 0.55, y + 0.16, w - 0.75, 0.26, font_name=BODY_FONT, font_size=14, color=PALETTE["light_text"], bold=True, margin=0)
        add_textbox(slide, body, x + 0.18, y + 0.52, w - 0.36, 0.82, font_name=BODY_FONT, font_size=14, color=PALETTE["light_text"], margin=0, line_spacing=1.1)

    add_panel(slide, 0.90, 4.65, 11.55, 1.10, PALETTE["ink_2"], line=PALETTE["ink_2"], radius=False)
    add_textbox(slide, "Current message to collaborators: the simulation pipeline is stable, the learned-LIF training setup is principled for sparse spikes, and voltage augmentation improves some metrics without replacing the spike-based interpretation.", 1.15, 4.97, 11.0, 0.42, font_name=BODY_FONT, font_size=16, color=PALETTE["light_text"], align=PP_ALIGN.CENTER, margin=0, line_spacing=1.15)
    add_network_motif(slide, 10.35, 0.90, scale=1.35)
    add_footer(slide, "Current simulation and learned-LIF workspace", 16, dark=True)


def build_dataset_truth_vs_correlation_slide(prs, dataset, slide_num):
    paths = dataset_figure_paths(dataset)
    summary = load_learned_parameter_summary(dataset)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["paper"])
    add_tag(slide, "CONNECTIVITY FIGURES", 0.65, 0.45, 2.10, 0.32, PALETTE["brick"], PALETTE["light_text"])
    add_textbox(slide, f'{dataset["label"]}: ground truth vs lagged correlation', 0.65, 0.88, 7.6, 0.54, font_name=TITLE_FONT, font_size=22, color=PALETTE["dark_text"], bold=True, margin=0)

    add_image_contain(slide, paths["true"], 0.70, 1.55, 6.05, 4.95, border_color=rgb("B7C3CE"), bg_color=PALETTE["sand"])
    add_image_contain(slide, paths["lagged"], 6.95, 1.55, 5.70, 4.95, border_color=rgb("B7C3CE"), bg_color=PALETTE["sand"])

    add_panel(slide, 0.85, 6.55, 11.80, 0.38, PALETTE["ink_2"], line=PALETTE["ink_2"], radius=False)
    add_textbox(
        slide,
        f'Lagged correlation shows the top {summary["top_k"]} undirected edges at {format_metric(summary["edge_cutoff"]) } cutoff using {summary["bin_ms"]:.0f} ms bins and a {summary["lag_window_ms"]:.0f} ms lag window.',
        1.05,
        6.64,
        11.4,
        0.18,
        font_name=BODY_FONT,
        font_size=12,
        color=PALETTE["light_text"],
        margin=0,
        align=PP_ALIGN.CENTER,
    )
    set_speaker_notes(slide, build_truth_slide_notes(dataset, summary))
    add_footer(slide, f'{dataset["label"]}: truth vs lagged correlation', slide_num)


def build_dataset_learned_lif_slide(prs, dataset, slide_num):
    paths = dataset_figure_paths(dataset)
    summary = load_learned_parameter_summary(dataset)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_background(slide, PALETTE["sand"])
    add_tag(slide, "LEARNED-LIF K=100", 0.65, 0.45, 1.95, 0.32, PALETTE["forest"], PALETTE["light_text"])
    add_textbox(slide, f'{dataset["label"]}: learned-LIF connectivity and learned parameters', 0.65, 0.88, 10.2, 0.54, font_name=TITLE_FONT, font_size=21, color=PALETTE["dark_text"], bold=True, margin=0)

    add_image_contain(slide, paths["learned"], 0.70, 1.55, 8.60, 4.72, border_color=rgb("B7C3CE"), bg_color=PALETTE["paper"])

    add_panel(slide, 9.55, 1.55, 3.00, 2.10, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Learned membrane parameters", 9.78, 1.78, 2.4, 0.22, font_name=BODY_FONT, font_size=13, color=PALETTE["teal"], bold=True, margin=0)
    add_textbox(
        slide,
        f'α = {summary["alpha"]:.4f}\nτ_m ≈ {summary["tau_ms"]:.1f} ms\nθ = {summary["threshold"]:.4f}\nβ = {summary["beta"]:.4f}\nreset = {summary["reset"]:.4f}',
        9.78,
        2.10,
        2.45,
        1.30,
        font_name=MONO_FONT,
        font_size=12,
        color=PALETTE["dark_text"],
        margin=0,
        line_spacing=1.0,
    )

    add_panel(slide, 9.55, 3.90, 3.00, 1.35, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "K=100 setup", 9.78, 4.12, 1.4, 0.20, font_name=BODY_FONT, font_size=13, color=PALETTE["forest"], bold=True, margin=0)
    add_textbox(
        slide,
        f'candidate mode: {summary["candidate_mode"]}\nK = {summary["k"]}, delay = {summary["max_delay"]}\nspatial/temporal = {summary["spatial_k"]}/{summary["temporal_k"]}',
        9.78,
        4.44,
        2.45,
        0.62,
        font_name=MONO_FONT,
        font_size=11,
        color=PALETTE["dark_text"],
        margin=0,
        line_spacing=1.0,
    )

    add_panel(slide, 9.55, 5.45, 3.00, 1.10, PALETTE["paper"], line=rgb("D8D1C5"), radius=False)
    add_textbox(slide, "Map metrics", 9.78, 5.64, 1.2, 0.20, font_name=BODY_FONT, font_size=13, color=PALETTE["brick"], bold=True, margin=0)
    add_textbox(
        slide,
        f'Learned-LIF AUC/AP: {format_metric(summary["learned_auc"])} / {format_metric(summary["learned_ap"])}\nLagged corr AUC/AP: {format_metric(summary["lagged_auc"])} / {format_metric(summary["lagged_ap"])}\n|W| threshold = {format_metric(summary["learned_threshold"])}; shown edges = {summary["learned_edges"]}',
        9.78,
        5.92,
        2.45,
        0.50,
        font_name=BODY_FONT,
        font_size=10,
        color=PALETTE["dark_text"],
        margin=0,
        line_spacing=1.0,
    )
    add_panel(slide, 0.70, 6.36, 8.60, 0.40, PALETTE["ink_2"], line=PALETTE["ink_2"], radius=False)
    add_textbox(slide, 'Takeaway', 0.92, 6.47, 0.90, 0.14, font_name=BODY_FONT, font_size=11, color=PALETTE["gold"], bold=True, margin=0)
    add_textbox(
        slide,
        build_dataset_takeaway_text(summary),
        1.78,
        6.43,
        7.20,
        0.20,
        font_name=BODY_FONT,
        font_size=10,
        color=PALETTE["light_text"],
        margin=0,
        line_spacing=1.0,
    )
    set_speaker_notes(slide, build_learned_slide_notes(dataset, summary))
    add_footer(slide, f'{dataset["label"]}: learned-LIF K=100', slide_num)


def verify_assets():
    missing = [str(path) for path in ASSETS.values() if not path.exists()]
    for dataset in CONNECTIVITY_DATASETS:
        figure_paths = dataset_figure_paths(dataset)
        required_paths = [
            figure_paths["true"],
            figure_paths["lagged"],
            figure_paths["learned"],
            figure_paths["correlation_npz"],
            dataset["learned_npz"],
            dataset["learned_pt"],
        ]
        missing.extend(str(path) for path in required_paths if not path.exists())
    if missing:
        raise FileNotFoundError("Missing presentation assets:\n  - " + "\n  - ".join(missing))


def build_presentation():
    verify_assets()

    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    prs.core_properties.title = "Clustered LIF Simulation and Learned-LIF Connectivity Inference"
    prs.core_properties.author = "GitHub Copilot"
    prs.core_properties.subject = "Simulation, preprocessing, training strategy, and learned-LIF results"

    build_title_slide(prs)
    build_simulation_slide(prs)
    build_preprocess_slide(prs)
    build_pipeline_slide(prs)
    build_training_slide(prs)
    build_model_slide(prs)
    build_voltage_slide(prs)
    build_results_slide(prs)
    build_dataset_truth_vs_correlation_slide(prs, CONNECTIVITY_DATASETS[0], 9)
    build_dataset_learned_lif_slide(prs, CONNECTIVITY_DATASETS[0], 10)
    build_dataset_truth_vs_correlation_slide(prs, CONNECTIVITY_DATASETS[1], 11)
    build_dataset_learned_lif_slide(prs, CONNECTIVITY_DATASETS[1], 12)
    build_dataset_truth_vs_correlation_slide(prs, CONNECTIVITY_DATASETS[2], 13)
    build_dataset_learned_lif_slide(prs, CONNECTIVITY_DATASETS[2], 14)
    build_tradeoff_slide(prs)
    build_close_slide(prs)

    prs.save(str(OUTPUT_PPTX))
    return OUTPUT_PPTX


if __name__ == "__main__":
    output_path = build_presentation()
    print(f"Saved presentation to: {output_path}")