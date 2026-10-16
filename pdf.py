"""
Generates a formatted PDF of the KV-Cache Continuity Master Experiment Execution Guide.
Fixed for Windows/Latin-1 compatibility.
Requires: pip install fpdf2
"""

from fpdf import FPDF

class ExperimentGuidePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, "KV-Cache Continuity Paper: Master Experiment Execution Guide", align="C")
            self.ln(5)
            
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title, level=1):
        # Replace any remaining unicode symbols with ASCII safe versions
        safe_title = title.replace("\u2022", "-").replace("\u2248", "~").replace("\u2208", "in")
        
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(0, 51, 102)
            self.ln(8)
            self.cell(0, 10, safe_title)
            self.ln(10)
            self.set_draw_color(0, 51, 102)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(0, 76, 153)
            self.ln(6)
            self.cell(0, 9, safe_title)
            self.ln(9)
        elif level == 3:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(50, 50, 50)
            self.ln(4)
            self.cell(0, 8, safe_title)
            self.ln(8)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        clean_text = text.replace("**", "").replace("\u2248", "~").replace("\u2208", "in")
        self.multi_cell(0, 5, clean_text)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        clean_text = text.replace("**", "").replace("\u2248", "~").replace("\u2208", "in")
        # Use ASCII hyphen instead of Unicode bullet to avoid Latin-1 encoding errors
        self.cell(5, 5, "-")
        self.multi_cell(0, 5, f" {clean_text}")
        self.ln(1)

    def code_block(self, code):
        self.set_font("Courier", "", 8)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        self.set_draw_color(200, 200, 200)
        
        lines = code.strip().split('\n')
        block_height = len(lines) * 4 + 6
        
        if self.get_y() + block_height > 277:
            self.add_page()
            
        y_start = self.get_y()
        self.rect(10, y_start, 190, block_height, style="DF")
        self.set_xy(12, y_start + 3)
        
        for line in lines:
            # Courier handles most ASCII code fine; strip rare unicode just in case
            safe_line = line.replace("\u2248", "~").replace("\u2208", "in")
            self.cell(0, 4, safe_line)
            self.ln(4)
            self.set_x(12)
        self.ln(4)

    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            n = len(headers)
            col_widths = [190 / n] * n
            
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in data:
            max_lines = max(len(str(c).split('\n')) for c in row) if row else 1
            row_h = 6 * max_lines
            
            if self.get_y() + row_h > 277:
                self.add_page()
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(0, 51, 102)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
                self.ln()
                self.set_font("Helvetica", "", 8)
                self.set_text_color(30, 30, 30)
                fill = False

            self.set_fill_color(245, 245, 245) if fill else self.set_fill_color(255, 255, 255)
            for i, cell_data in enumerate(row):
                safe_data = str(cell_data).replace("\u2248", "~")
                self.cell(col_widths[i], row_h, safe_data, border=1, fill=True, align="L" if i == 0 else "C")
            self.ln()
            fill = not fill
        self.ln(4)


def build_pdf():
    pdf = ExperimentGuidePDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title Page
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 51, 102)
    pdf.ln(40)
    pdf.cell(0, 15, "Master Experiment Execution Guide", align="C")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "KV-Cache Continuity and Intermediate Representation Limits", align="C")
    pdf.ln(10)
    pdf.cell(0, 10, "in Depth-Truncated Transformers", align="C")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 12)
    pdf.cell(0, 8, "Reviewer Revisions | Missing Baselines | Rigor Diagnostics | Speculative Decoding", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "Target: Kaggle (P100/T4) | Colab | Local CUDA/CPU", align="C")

    # Section 0
    pdf.add_page()
    pdf.section_title("0. Four Critical Rules (Read Before Spending GPU Hours)")
    
    rules = [
        ("Joint Condition Needs Matching Adapters (Fix 1)", 
         "Never evaluate the joint condition using an adapter trained at a single depth (L=18) or on an out-of-domain dataset (TinyStories). Train the multi-depth in-domain adapter covering all candidate depths L in {18, 22, 24, 30} on WikiText-2 first (Phase 3), then evaluate the joint condition at those identical depths."),
        ("Calibrated Hypothesis Phrasing (Fix 2)", 
         'If the multi-depth adapter reaches an empirical plateau near ~82% Top-1 agreement, do NOT claim an "intrinsic manifold limitation". Instead, report it precisely as: "An empirical representation ceiling for this specific adapter family (bottleneck MLP with residual connection) under the given parameter budget (r=512), rather than an absolute theoretical bound."'),
        ("Strict Precision & Dtype Consistency (Fix 3)", 
         "NVIDIA Tesla P100 GPUs have NO native bfloat16 hardware support, and T4 GPUs emulate bfloat16 slowly. Always pass --dtype float16 across all scripts on Kaggle. NEVER mix dtypes within the same table. Every row in a comparison table must share identical precision."),
        ("Gated Llama-3.2 Authentication (Fix 4)", 
         "meta-llama/Llama-3.2-1B is a gated Hugging Face repository. Add your HF access token as a Kaggle Secret named HF_TOKEN and ensure Notebook Settings -> Internet is turned ON.")
    ]
    for title, desc in rules:
        pdf.section_title(title, level=3)
        pdf.body_text(desc)

    # Section 1
    pdf.section_title("1. Environment Setup & Pre-Flight Smoke Checks")
    pdf.section_title("1.1. Kaggle / Colab Initial Setup", level=2)
    pdf.code_block("""# Verify GPU accelerator
!nvidia-smi

# Install required dependencies
!pip install -q torch transformers datasets accelerate sentencepiece

# Authenticate Hugging Face for Llama-3.2
import os
from huggingface_hub import login
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
    print("[OK] Hugging Face authenticated successfully.")
else:
    print("[!] Warning: HF_TOKEN secret not found.")""")

    pdf.body_text("Set repository directory and run unit tests:")
    pdf.code_block("""%cd /kaggle/working/kv_continuity
export PYTHONPATH=.
!python -m unittest discover tests""")

    pdf.section_title("1.2. Pre-Flight Smoke Test Suite (< 3 minutes)", level=2)
    pdf.code_block("""# Smoke test Table 1 headline script
python experiments/run_tier1_interventions.py --model-type qwen \\
  --model-id Qwen/Qwen2.5-3B --dtype float16 --num-samples 2 \\
  --seeds 42 --sweep-depths 18 --skip-swap --skip-cosine \\
  --device cuda --output-dir results/smoke_test

# Smoke test Llama shallow sweep
python experiments/run_multiseed_llama_sweep.py \\
  --model-id meta-llama/Llama-3.2-1B --depths 4 --seeds 42 \\
  --num-samples 2 --device cuda --output-dir results/smoke_test

# Smoke test Causal KV swap
python experiments/run_robust_kv_swap_and_metrics.py \\
  --model-id Qwen/Qwen2.5-3B --depths 18 --num-samples 2 \\
  --device cuda --output-dir results/smoke_test

# Smoke test Autoregressive generation
python experiments/run_autoregressive_generation.py \\
  --model-id Qwen/Qwen2.5-3B --depths 18 --num-samples 2 \\
  --max-new-tokens 16 --device cuda --output-dir results/smoke_test

# Smoke test Extra fill baselines
python experiments/run_kv_baselines.py \\
  --model-id Qwen/Qwen2.5-3B --depths 18 --num-samples 2 \\
  --device cuda --output-dir results/smoke_test""")

    # Section 2
    pdf.add_page()
    pdf.section_title("2. Priority Phase 1: Core Reviewer Objections")
    
    pdf.section_title("2.1. Standardized Table I (N=30, 3 Seeds) with Cache Omission (A1)", level=2)
    pdf.bullet_point("Goal: Standardizes headline eval on WikiText-2 across N=30 sequences with 3 seeds (42, 123, 7).")
    pdf.bullet_point("Runtime: ~12-15 mins on T4/P100; ~4 mins on A100.")
    pdf.code_block("""python experiments/run_tier1_interventions.py \\
  --model-type qwen --model-id Qwen/Qwen2.5-3B --dtype float16 \\
  --num-samples 30 --seeds 42,123,7 --sweep-depths 18,24,30 \\
  --skip-swap --skip-cosine --device cuda \\
  --output-dir results/canonical_sweeps/revision_table1""")

    pdf.section_title("2.4. Shallow Llama-3.2-1B Depth Sweep (A5)", level=2)
    pdf.bullet_point("Goal: Sweeps shallow exit depths on Llama-3.2-1B across 3 seeds to pinpoint crossing point below ln(2) ~ 0.6931 nats.")
    pdf.code_block("""python experiments/run_multiseed_llama_sweep.py \\
  --model-id meta-llama/Llama-3.2-1B --depths 4,5,6,7,8,10,12,14 \\
  --seeds 42,123,7 --num-samples 30 --device cuda \\
  --output-dir results/canonical_sweeps/revision_llama_shallow""")

    pdf.section_title("2.3. Centered Cosine Similarity & Direct Attn-Output Error (A10)", level=2)
    pdf.code_block("""python experiments/run_tier1_interventions.py \\
  --model-type qwen --model-id Qwen/Qwen2.5-3B --dtype float16 \\
  --swap-depths 18 --num-samples 30 --skip-table1 --skip-swap \\
  --device cuda --output-dir results/canonical_sweeps/revision_cosine_attn""")

    pdf.section_title("2.2. Expanded Causal Key/Value Swap Across Depths (A9)", level=2)
    pdf.code_block("""# Qwen2.5-3B (Depths 14, 18, 22, 24, 30):
python experiments/run_robust_kv_swap_and_metrics.py \\
  --model-id Qwen/Qwen2.5-3B --depths 14,18,22,24,30 \\
  --num-samples 30 --device cuda \\
  --output-dir results/canonical_sweeps/revision_kv_swap_qwen

# Llama-3.2-1B (Depths 6, 8, 10):
python experiments/run_robust_kv_swap_and_metrics.py \\
  --model-id meta-llama/Llama-3.2-1B --depths 6,8,10 \\
  --num-samples 30 --device cuda \\
  --output-dir results/canonical_sweeps/revision_kv_swap_llama""")

    # Section 3
    pdf.add_page()
    pdf.section_title("3. Priority Phase 2: Autoregressive Generation & Extra Baselines")
    
    pdf.section_title("4.1. Autoregressive (Non-Teacher-Forced) Generation (A3)", level=2)
    pdf.bullet_point("Goal: Measures compounding error in real open-loop AR generation over 256 tokens.")
    pdf.code_block("""python experiments/run_autoregressive_generation.py \\
  --model-id Qwen/Qwen2.5-3B --depths 18,24,30 --num-samples 30 \\
  --max-new-tokens 256 --device cuda \\
  --output-dir results/canonical_sweeps/revision_autoregressive""")

    pdf.section_title("4.2. Extra Fill Baselines (A2, A6)", level=2)
    pdf.code_block("""python experiments/run_kv_baselines.py \\
  --model-id Qwen/Qwen2.5-3B --depths 18,24,30 --num-samples 30 \\
  --device cuda --output-dir results/canonical_sweeps/revision_kv_baselines""")

    # Section 4
    pdf.section_title("4. Priority Phase 3: Multi-Depth Adapter Training & Joint Condition")
    
    pdf.section_title("3.2. Multi-Depth In-Domain Adapter Training (A12)", level=2)
    pdf.bullet_point("Capacity Test: Run 10,000 steps first (~10M tokens). Continue to 25k if GPU hours allow.")
    pdf.bullet_point("Runtime (10k steps): ~2.5 hours on T4/P100; ~45 mins on A100.")
    pdf.code_block("""python experiments/train_exit_adapters.py \\
  --model-type qwen --model-id Qwen/Qwen2.5-3B \\
  --candidate-depths 18,22,24,30 --adapter-type mlp --rank 512 \\
  --steps 10000 --batch-size 8 --seq-len 128 --eval-every 500 \\
  --device cuda --output-dir checkpoints/multi_depth_r512""")

    pdf.section_title("3.1. Joint Early-Exit Condition Evaluation (A4)", level=2)
    pdf.code_block("""python experiments/run_joint_condition_sweep.py \\
  --model-id Qwen/Qwen2.5-3B --depths 18,22,24,30 \\
  --adapter-path checkpoints/multi_depth_r512/final_exit_adapters_qwen_mlp_r512.pt \\
  --num-samples 30 --device cuda \\
  --output-dir results/canonical_sweeps/revision_joint_condition""")

    # Section 5
    pdf.add_page()
    pdf.section_title("5. Priority Phase 4: Extended Generalization, Robustness & Downstream")
    
    pdf.section_title("5.1. Long-Context Scaling (T in {1024, 2048, 4096}) (A8)", level=2)
    pdf.code_block("""python experiments/run_long_context_scaling.py \\
  --model-id Qwen/Qwen2.5-3B --depths 18,24,30 --device cuda \\
  --output-dir results/canonical_sweeps/revision_long_context""")

    pdf.section_title("5.2. Downstream Zero-Shot Benchmarks (A7)", level=2)
    pdf.code_block("""python experiments/run_downstream_benchmarks.py \\
  --model-id Qwen/Qwen2.5-3B --depths 22,24,30 --device cuda \\
  --output-dir results/canonical_sweeps/revision_downstream""")

    pdf.section_title("5.3. Entropy & Domain Breakdown (A11)", level=2)
    pdf.code_block("""python experiments/run_entropy_domain_comparison.py \\
  --model-id Qwen/Qwen2.5-3B --depths 18,24 --device cuda \\
  --output-dir results/canonical_sweeps/revision_entropy""")

    pdf.section_title("5.4. Large Model Scaling (7B-8B with Quantization) (A13)", level=2)
    pdf.code_block("""python experiments/run_large_model_sweep.py \\
  --model-id Qwen/Qwen2.5-7B --depths 14,20,24 --load-in-8bit \\
  --device cuda --output-dir results/canonical_sweeps/revision_7b""")

    pdf.section_title("5.5. Active Speculative Decoding with Cascading Reads (A14)", level=2)
    pdf.code_block("""python experiments/run_active_speculative_decoding.py \\
  --model-id Qwen/Qwen2.5-3B --d1 18 --d2 24 --device cuda \\
  --output-dir results/canonical_sweeps/revision_active_spec""")

    # Section 6
    pdf.add_page()
    pdf.section_title("6. Priority Phase 5: Scheduled Speculative Decoding & Tuned-Lens")
    
    pdf.section_title("6.1. Multi-Token Speculative Smoke Benchmark (gamma=2)", level=2)
    pdf.code_block("""python experiments/benchmark_speculative_decoding.py \\
  --model-id Qwen/Qwen2.5-3B \\
  --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \\
  --draft-depth 24 --gamma 2 --max-tokens 40 --dtype float16 --device cuda""")

    pdf.section_title("6.2. Fast Tuned-Lens Calibration Training (< 3 Mins)", level=2)
    pdf.body_text("Exploits Key-stable / Value-evolving asymmetry via ultra-lightweight rank-64 low-rank affine tuned lens.")
    pdf.code_block("""python experiments/train_tuned_lens.py \\
  --model-id Qwen/Qwen2.5-3B --target-depths 14,18,22,24,28 \\
  --adapter-type lora --rank 64 --steps 400 --batch-size 2 \\
  --seq-len 128 --lr 1e-3 --distill-temp 2.0 \\
  --output-path checkpoints/exit_adapters_qwen_tuned_lens.pt --device cuda""")

    pdf.section_title("6.3. Systematic Speculative Depth Sweep", level=2)
    pdf.code_block("""# Evaluate baseline MLP adapter:
python experiments/sweep_speculative_depths.py \\
  --model-id Qwen/Qwen2.5-3B \\
  --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \\
  --depths 14,18,22,24,28,30 --gamma 1 --num-prompts 12 \\
  --max-tokens 30 --dtype float16 --device cuda \\
  --output-dir results/canonical_sweeps/qwen2.5_3b

# Evaluate newly calibrated Tuned-Lens adapter:
python experiments/sweep_speculative_depths.py \\
  --model-id Qwen/Qwen2.5-3B \\
  --adapter-path checkpoints/exit_adapters_qwen_tuned_lens.pt \\
  --depths 14,18,22,24,28,30 --gamma 1 --num-prompts 12 \\
  --max-tokens 30 --dtype float16 --device cuda \\
  --output-dir results/canonical_sweeps/qwen2.5_3b""")

    pdf.section_title("6.4. Publication-Scale 60-Prompt Multi-Domain Benchmark", level=2)
    pdf.code_block("""python experiments/benchmark_large_scale.py \\
  --model-id Qwen/Qwen2.5-3B \\
  --adapter-path checkpoints/exit_adapters_qwen_tuned_lens.pt \\
  --draft-depth 24 --gamma 1 --num-prompts-per-domain 15 \\
  --max-tokens 30 --device cuda""")

    # Section 7
    pdf.add_page()
    pdf.section_title("7. Wall-Clock Runtime Summary")
    
    time_headers = ["Phase", "Description", "Est. Time (T4/P100)", "Est. Time (A100)"]
    time_data = [
        ["Pre-Flight", "Unit Tests & Smoke Tests", "~2 - 3 mins", "~1 min"],
        ["Phase 1", "Table 1, Llama, Cosine, Swap", "~45 - 55 mins", "~15 - 18 mins"],
        ["Phase 2", "AR Gen & Extra Fill Baselines", "~25 - 32 mins", "~8 - 10 mins"],
        ["Phase 3", "Adapter Train (10k) & Joint", "~2.5 - 3.0 hrs", "~45 - 55 mins"],
        ["Phase 4", "Long-Ctx, Downstream, 7B, Spec", "~60 - 75 mins", "~20 - 25 mins"],
        ["Phase 5", "Spec Decode & Tuned-Lens Track", "~25 - 30 mins", "~8 - 10 mins"],
        ["TOTAL", "Entire KV Paper Evidence Suite", "~5.0 - 6.0 hrs", "~1.8 - 2.2 hrs"],
    ]
    pdf.add_table(time_headers, time_data, col_widths=[25, 75, 45, 45])

    pdf.section_title("7.2. Recommended Two-Session Kaggle Plan", level=2)
    pdf.body_text("Session 1: All Inference & Evaluation Experiments (~1.5 to 2 hours). Execute Phases 1, 2, 4, and 5. Zip and download results.")
    pdf.code_block("""import shutil
shutil.make_archive("kv_inference_results", "zip", "results")
print("[OK] Session 1 results ready for download!")""")
    
    pdf.body_text("Session 2: Adapter Training & Joint Condition (~2.5 to 3 hours). Execute Phase 3. Zip and download checkpoints.")
    pdf.code_block("""import shutil
shutil.make_archive("kv_adapter_checkpoints", "zip", "checkpoints")
shutil.make_archive("kv_joint_results", "zip", "results")
print("[OK] Session 2 checkpoints and results ready!")""")

    output_path = "KV_Cache_Continuity_Experiment_Guide.pdf"
    pdf.output(output_path)
    print(f"[SUCCESS] PDF generated successfully: {output_path}")


if __name__ == "__main__":
    build_pdf()