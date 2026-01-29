import argparse
import os
import sys
import torch
import subprocess
import shutil
import json
import traceback
from torchinfo import summary

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- PlotNeuralNet Helpers ---

def setup_pnn_styles(base_dir, target_dir):
    """Copy PlotNeuralNet style files and nn_blocks.tex to the target directory."""
    vendor_layer_dir = os.path.join(base_dir, 'vendor', 'PlotNeuralNet', 'layers')
    styles = ['Ball.sty', 'Box.sty', 'RightBandedBox.sty']
    
    if not os.path.exists(vendor_layer_dir):
        print(f"Warning: PlotNeuralNet vendor directory not found at {vendor_layer_dir}")
        return

    # Copy Styles
    for s in styles:
        src = os.path.join(vendor_layer_dir, s)
        dst = os.path.join(target_dir, s)
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Failed to copy {s}: {e}")
        else:
            print(f"Warning: Style file {src} does not exist.")
            
    # Copy nn_blocks.tex
    src_blocks = os.path.join(base_dir, 'nn_blocks.tex')
    dst_blocks = os.path.join(target_dir, 'nn_blocks.tex')
    if os.path.exists(src_blocks):
        try:
            shutil.copy2(src_blocks, dst_blocks)
        except Exception as e:
             print(f"Failed to copy nn_blocks.tex: {e}")

def get_pnn_header():
    return r"""
\definecolor{convcolor}{rgb}{1,0.8,0.5}
\definecolor{rescolor}{rgb}{0.8,0.8,1}
\definecolor{poolcolor}{rgb}{1,0.5,0.5}
\definecolor{upcolor}{rgb}{0.8,0.5,1}
\definecolor{opcolor}{rgb}{0.5,1,0.5}
"""

# --- Generators ---

def gen_tikz_video_swin(info, out_tex):
    header = get_pnn_header()
    tex = f"""
\\documentclass[tikz, border=2mm]{{standalone}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning, calc, 3d}}
\\input{{nn_blocks.tex}}
{header}
\\def\\ConvColor{{convcolor}}
\\def\\ResColor{{rescolor}}
\\def\\OpColor{{opcolor}}

\\begin{{document}}
\\begin{{tikzpicture}}

\\node[io] (in) {{Input Video\\\\(T x H x W)}};

% Patch Embed (3D Conv)
\pic[shift={{(2.5,0,0)}}] at (in.east) {{RightBandedBox={{
    name=pe,
    caption=3D Patch Embed,
    xlabel={{"Dimensions"}},
    fill=\\ConvColor,
    height=30, width=4, depth=30
}}}};

% Swin Block 1
\\pic[shift={{(2.0,0,0)}}] at (pe-east) {{Box={{
    name=blk1,
    caption=Swin\\\\Block 1,
    fill=\\ResColor,
    height=30, width=6, depth=30
}}}};

% Swin Block 2
\\pic[shift={{(1.5,0,0)}}] at (blk1-east) {{Box={{
    name=blk2,
    caption=Swin\\\\Block 2,
    fill=\\ResColor,
    height=30, width=6, depth=30
}}}};

% Output Projection
\\pic[shift={{(2.0,0,0)}}] at (blk2-east) {{RightBandedBox={{
    name=proj,
    caption=Output\\\\Proj,
    fill=\\ConvColor,
    height=30, width=2, depth=30
}}}};

\\node[io, right=2.0cm of proj-east] (out) {{Prediction}};

% Connections
\\draw[arrow] (in) -- (pe-west);
\\draw[arrow] (pe-east) -- (blk1-west);
\\draw[arrow] (blk1-east) -- (blk2-west);
\\draw[arrow] (blk2-east) -- (proj-west);
\\draw[arrow] (proj-east) -- (out);

\\end{{tikzpicture}}
\\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_conv_lstm(info, out_tex):
    header = get_pnn_header()
    tex = f"""
\\documentclass[tikz, border=2mm]{{standalone}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning, calc, 3d}}
\\input{{nn_blocks.tex}}
{header}
\\def\\ConvColor{{convcolor}}
\\def\\ResColor{{rescolor}}
\\def\\OpColor{{opcolor}}

\\begin{{document}}
\\begin{{tikzpicture}}

\\node[io] (in) {{Input Seq}};

% Layer 1 Cell
\\pic[shift={{(2.0,0,0)}}] at (in.east) {{Box={{
    name=cell1,
    caption=ConvLSTM\\\\Layer 1,
    fill=\\ConvColor,
    height=25, width=5, depth=25
}}}};

% Recurrent Loop 1
\\draw[skip] (cell1-east) -- ++(0.5, 0, 0) -- ++(0, 3, 0) -- ++(-6, 0, 0) |- (cell1-west);

% Layer 2 Cell
\\pic[shift={{(2.0,0,0)}}] at (cell1-east) {{Box={{
    name=cell2,
    caption=ConvLSTM\\\\Layer 2,
    fill=\\ConvColor,
    height=25, width=5, depth=25
}}}};

% Recurrent Loop 2
\\draw[skip] (cell2-east) -- ++(0.5, 0, 0) -- ++(0, 3, 0) -- ++(-6, 0, 0) |- (cell2-west);

% Output Proj
\\pic[shift={{(2.0,0,0)}}] at (cell2-east) {{RightBandedBox={{
    name=proj,
    caption=Proj,
    fill=\\OpColor,
    height=25, width=2, depth=25
}}}};

\\node[io, right=2.0cm of proj-east] (out) {{Pred}};

\\draw[arrow] (in) -- (cell1-west);
\\draw[arrow] (cell1-east) -- (cell2-west);
\\draw[arrow] (cell2-east) -- (proj-west);
\\draw[arrow] (proj-east) -- (out);

\\end{{tikzpicture}}
\\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_physics_transformer(info, out_tex):
    header = get_pnn_header()
    tex = f"""
\\documentclass[tikz, border=2mm]{{standalone}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning, calc, 3d}}
\\input{{nn_blocks.tex}}
{header}
\\def\\ConvColor{{convcolor}}
\\def\\ResColor{{rescolor}}
\\def\\OpColor{{opcolor}}

\\begin{{document}}
\\begin{{tikzpicture}}

\\node[io] (in) {{Input}};

% Input Proj
\\pic[shift={{(2.0,0,0)}}] at (in.east) {{Box={{
    name=proj_in,
    caption=Input\\\\Proj,
    fill=\\ConvColor,
    height=20, width=2, depth=20
}}}};

% Pos Encoding
\\pic[shift={{(1.5,0,0)}}] at (proj_in-east) {{Box={{
    name=pe,
    caption=Physics\\\\PosEnc,
    fill=\\OpColor,
    height=20, width=2, depth=20
}}}};

% Transformer Layers
\\pic[shift={{(2.0,0,0)}}] at (pe-east) {{Box={{
    name=layer1,
    caption=Physics\\\\Attn,
    fill=\\ResColor,
    height=20, width=4, depth=20
}}}};

\\pic[shift={{(1.0,0,0)}}] at (layer1-east) {{Box={{
    name=ffn1,
    caption=FFN,
    fill=\\ResColor,
    height=20, width=4, depth=20
}}}};

\\pic[shift={{(1.5,0,0)}}] at (ffn1-east) {{Box={{
    name=layer2,
    caption=Physics\\\\Attn,
    fill=\\ResColor,
    height=20, width=4, depth=20
}}}};

\\pic[shift={{(1.0,0,0)}}] at (layer2-east) {{Box={{
    name=ffn2,
    caption=FFN,
    fill=\\ResColor,
    height=20, width=4, depth=20
}}}};

% Output Proj
\\pic[shift={{(2.0,0,0)}}] at (ffn2-east) {{Box={{
    name=proj_out,
    caption=Output\\\\Proj,
    fill=\\ConvColor,
    height=20, width=2, depth=20
}}}};

\\node[io, right=2.0cm of proj_out-east] (out) {{Output}};

\\draw[arrow] (in) -- (proj_in-west);
\\draw[arrow] (proj_in-east) -- (pe-west);
\\draw[arrow] (pe-east) -- (layer1-west);
\\draw[arrow] (layer1-east) -- (ffn1-west);
\\draw[arrow] (ffn1-east) -- (layer2-west);
\\draw[arrow] (layer2-east) -- (ffn2-west);
\\draw[arrow] (ffn2-east) -- (proj_out-west);
\\draw[arrow] (proj_out-east) -- (out);

\\end{{tikzpicture}}
\\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_sequential(info, out_tex):
    header = get_pnn_header()
    tex = f"""
\\documentclass[tikz, border=2mm]{{standalone}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning, calc, 3d}}
\\input{{nn_blocks.tex}}
{header}
\\def\\ConvColor{{convcolor}}
\\def\\ResColor{{rescolor}}
\\def\\OpColor{{opcolor}}

\\begin{{document}}
\\begin{{tikzpicture}}

\\node[io] (in) {{Input Seq}};

% Stage 1: Spatial
\pic[shift={{(2.5,0,0)}}] at (in.east) {{RightBandedBox={{
    name=spatial,
    caption=Spatial\\\\Module,
    xlabel={{"Extract", }},
    fill=\\ConvColor,
    height=30, width=6, depth=30
}}}};

% Intermediate
\node[below=0.5cm of spatial-south] (inter) {{Spatial Features + Preds}};

% Stage 2: Temporal
\pic[shift={{(2.5,0,0)}}] at (spatial-east) {{RightBandedBox={{
    name=temporal,
    caption=Temporal\\\\Module,
    xlabel={{"Predict", }},
    fill=\\ResColor,
    height=30, width=6, depth=30
}}}};

\\node[io, right=2.5cm of temporal-east] (out) {{Final Pred}};

\\draw[arrow] (in) -- (spatial-west);
\\draw[arrow] (spatial-east) -- (temporal-west);
\\draw[arrow] (temporal-east) -- (out);

\\end{{tikzpicture}}
\\end{{document}}
"""
    out_tex.write(tex)

def process_model(model_name: str, args, report_data):
    print(f"\\nProcessing model: {model_name}...")
    entry = {
        'status': 'failed',
        'generator': 'generic',
        'paths': {},
        'error': ''
    }
    
    try:
        # Determine paths
        base_dir = os.path.dirname(__file__)
        export_dir = os.path.join(base_dir, 'build_export')
        
        # Create per-model subdirectory
        model_out_dir = os.path.join(export_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        
        # Setup styles
        setup_pnn_styles(base_dir, model_out_dir)
        
        # TikZ Generation
        tex_path = os.path.join(model_out_dir, f"fig_{model_name.lower()}_auto.tex")
        info = {'img': args.img}
        
        model_key = model_name.lower()
        with open(tex_path, "w") as f:
            if 'videoswin' in model_key:
                gen_tikz_video_swin(info, f)
                entry['generator'] = 'video_swin'
            elif 'convlstm' in model_key or 'conv_rnn' in model_key:
                gen_tikz_conv_lstm(info, f)
                entry['generator'] = 'conv_lstm'
            elif 'physics' in model_key:
                gen_tikz_physics_transformer(info, f)
                entry['generator'] = 'physics_transformer'
            elif 'sequential' in model_key:
                gen_tikz_sequential(info, f)
                entry['generator'] = 'sequential'
            else:
                raise ValueError(f"Unknown temporal model: {model_name}")
                
        entry['paths']['tex'] = tex_path
        
        # Compilation
        if args.compile:
            cmd = ['conda', 'run', '-n', args.latex_env, 'tectonic', tex_path]
            print(f"Compiling: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            
            pdf_path = os.path.join(model_out_dir, f"fig_{model_name.lower()}_auto.pdf")
            if os.path.exists(pdf_path):
                entry['paths']['pdf'] = pdf_path
                entry['status'] = 'success'
            else:
                entry['error'] = 'PDF file not found after compilation'
        else:
             entry['status'] = 'success'
             
    except Exception as e:
        print(f"FAILED {model_name}: {e}")
        entry['error'] = str(e)
        traceback.print_exc()
        
    report_data[model_name] = entry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="VideoSwin,ConvLSTM,PhysicsTransformer,Sequential", help="Comma-separated list of models")
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--compile", action="store_true", help="Compile generated TeX to PDF")
    ap.add_argument("--latex_env", type=str, default="latex")
    args = ap.parse_args()

    # Ensure style files are present
    base_dir = os.path.dirname(__file__)
    export_dir = os.path.join(base_dir, 'build_export')
    os.makedirs(export_dir, exist_ok=True)

    model_list = [m.strip() for m in args.models.split(',')]
    report_data = {}
    
    for m in model_list:
        process_model(m, args, report_data)
        
    report_path = os.path.join(base_dir, 'build_export', 'report_temporal.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\\nBatch processing complete. Report saved to {report_path}")
    print("\\nSummary:")
    for m, data in report_data.items():
        print(f"{m}: {data['status'].upper()} ({data['generator']})")

if __name__ == "__main__":
    main()
