import argparse
import os
import sys
import subprocess
import shutil
import traceback

def get_nntikz_header():
    return r"""
\documentclass[tikz, border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, chains, shapes.geometric, fit, shapes, arrows.meta, calc, backgrounds}

\begin{document}
\begin{tikzpicture}[
    >=LaTeX, 
    very thick,
    node distance=1.0cm and 1.2cm,
    arrow/.style={
        ->,
        very thick,
        rounded corners=0.2cm
    },
    block/.style={
        rectangle,
        fill=gray!10,
        rounded corners=3mm,
        draw,
        very thick,
        inner sep=0.8em
    },
    layer/.style={
        rectangle,
        fill=white!10,
        rounded corners=1mm,
        inner xsep=0.6em,
        inner ysep=0.6em,
        minimum height=3.0em,
        align=center,
        draw,
        very thick
    },
    conv/.style={
        layer,
        fill=orange!10,
        draw=orange!80!black
    },
    relu/.style={
        layer,
        fill=yellow!10,
        draw=yellow!80!black,
        minimum height=2.5em
    },
    pool/.style={
        layer,
        fill=red!10,
        draw=red!80!black
    },
    input/.style={ 
        circle,
        minimum width=3.0em,
        draw,
        fill=gray!20,
        thick,
        align=center
    },
    sum/.style={
        circle,
        draw,
        fill=white,
        inner sep=0pt,
        minimum size=1.2em,
        node contents={+}
    }
]
"""

def gen_tikz_edsr_2d(out_tex):
    header = get_nntikz_header()
    tex = rf"""{header}

    % Input
    \node[input] (in) {{Input\\LR}};

    % Head
    \node[conv, right=of in] (head) {{Conv\\$3\times 3$}};
    \draw[arrow] (in) -- (head);

    % ResBlock Detail
    \node[conv, right=2.0cm of head] (res1_c1) {{Conv\\$3\times 3$}};
    \node[relu, right=0.8cm of res1_c1] (res1_act) {{ReLU}};
    \node[conv, right=0.8cm of res1_act] (res1_c2) {{Conv\\$3\times 3$}};
    
    % Scaling constant
    \node[circle, draw, right=0.8cm of res1_c2, inner sep=1pt] (mul) {{$\times 0.1$}};
    
    % Sum
    \node (sum1) [sum, right=0.8cm of mul];

    % ResBlock Box
    \begin{{scope}}[on background layer]
        \node[block, fit=(res1_c1) (sum1), label=above:ResBlock (Detailed)] (resblock) {{}};
    \end{{scope}}

    % Connections inside ResBlock
    \draw[arrow] (head) -- (res1_c1);
    \draw[arrow] (res1_c1) -- (res1_act);
    \draw[arrow] (res1_act) -- (res1_c2);
    \draw[arrow] (res1_c2) -- (mul);
    \draw[arrow] (mul) -- (sum1);
    
    % Skip connection
    \draw[arrow] (head) -- ++(0.8,0) |- ($(resblock.north) + (0, 0.8)$) -| (sum1.north);

    % More Blocks (Abstract)
    \node[right=1.2cm of resblock] (dots) {{\huge $\dots$}};
    \draw[arrow] (sum1) -- (dots);

    \node[block, right=1.2cm of dots, minimum height=3em, align=center] (resN) {{ResBlock\\$\times 16$}};
    \draw[arrow] (dots) -- (resN);

    % Tail
    \node[conv, right=of resN] (tail) {{Conv}};
    \draw[arrow] (resN) -- (tail);

    % Global Skip
    \draw[arrow] (head) -- ++(0.8,0) |- ($(resblock.south) + (0, -1.5)$) -| (tail.south);

    % Upsampler
    \node[layer, right=of tail, fill=purple!10] (up) {{PixelShuffle\\$\times 4$}};
    \draw[arrow] (tail) -- (up);

    % Output
    \node[input, right=of up] (out) {{Output\\HR}};
    \draw[arrow] (up) -- (out);

\end{{tikzpicture}}
\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_unet_2d(out_tex):
    header = get_nntikz_header()
    tex = rf"""{header}

    % Input
    \node[input] (in) {{Input}};

    % Encoder 1 (Detailed)
    \node[conv, right=of in] (enc1_c1) {{Conv\\$3\times 3$}};
    \node[relu, right=0.5cm of enc1_c1] (enc1_r1) {{ReLU}};
    \node[conv, right=0.5cm of enc1_r1] (enc1_c2) {{Conv\\$3\times 3$}};
    \node[relu, right=0.5cm of enc1_c2] (enc1_r2) {{ReLU}};
    
    \begin{{scope}}[on background layer]
        \node[block, fit=(enc1_c1) (enc1_r2), label=above:Encoder 1] (enc1) {{}};
    \end{{scope}}
    
    \draw[arrow] (in) -- (enc1_c1);
    \draw[arrow] (enc1_c1) -- (enc1_r1);
    \draw[arrow] (enc1_r1) -- (enc1_c2);
    \draw[arrow] (enc1_c2) -- (enc1_r2);

    % Pool 1 (Aligned below enc1_r2 to avoid backtracking)
    \node[pool, below=1.5cm of enc1_r2] (pool1) {{MaxPool}};
    % Path from rightmost encoder node to pool
    \draw[arrow] (enc1_r2.east) -- ++(0.5,0) |- (pool1.north);

    % Encoder 2 (Aligned with Encoder 1 end)
    \node[layer, below=1.5cm of pool1] (enc2) {{DoubleConv\\(Enc 2)}};
    \draw[arrow] (pool1) -- (enc2);

    % Bottleneck
    \node[pool, below=1.5cm of enc2] (pool2) {{MaxPool}};
    \draw[arrow] (enc2) -- (pool2);
    
    \node[layer, below=1.5cm of pool2, fill=blue!10] (bot) {{Bottleneck\\DoubleConv}};
    \draw[arrow] (pool2) -- (bot);

    % --- Decoder Side (Strict Alignment) ---

    % Up2 (Same level as Bot, shifted right)
    \node[layer, right=3.5cm of bot, fill=green!10] (up2) {{UpConv}};
    \draw[arrow] (bot) -- (up2);

    % Dec2 (Aligned with Enc2)
    \node[layer, at={{(up2 |- enc2)}}] (dec2) {{DoubleConv\\(Dec 2)}};
    \draw[arrow] (up2) -- (dec2);

    % Skip Connection 2 (Horizontal)
    \draw[arrow, dashed] (enc2.east) -- node[above]{{Concat}} (dec2.west);

    % Up1 (Aligned with Pool1 level)
    \node[layer, at={{(dec2 |- pool1)}}, fill=green!10] (up1) {{UpConv}};
    \draw[arrow] (dec2) -- (up1);

    % Dec1 (Aligned with Enc1 output / Enc1 level)
    \node[layer, at={{(up1 |- enc1)}}] (dec1) {{DoubleConv\\(Dec 1)}};
    \draw[arrow] (up1) -- (dec1);

    % Skip Connection 1 (Horizontal-ish)
    \draw[arrow, dashed] (enc1.east) -- (dec1.west);

    % Output
    \node[conv, right=1.2cm of dec1] (out_conv) {{Conv\\$1\times 1$}};
    \node[input, right=of out_conv] (out) {{Output}};
    
    \draw[arrow] (dec1) -- (out_conv);
    \draw[arrow] (out_conv) -- (out);

\end{{tikzpicture}}
\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_fno2d_2d(out_tex):
    header = get_nntikz_header()
    tex = rf"""{header}

    \node[input] (in) {{Input}};
    
    % Lift
    \node[conv, right=of in] (lift) {{Lift\\(1x1 Conv)}};
    \draw[arrow] (in) -- (lift);

    % Fourier Layer Detail
    \node[layer, right=2.0cm of lift, fill=blue!5] (fft) {{FFT\\(2D)}};
    \node[layer, right=1.0cm of fft, fill=yellow!10] (weights) {{Spectral\\Transform\\$R + iI$}};
    \node[layer, right=1.0cm of weights, fill=blue!5] (ifft) {{IFFT\\(2D)}};
    
    % Skip (W)
    \node[conv, above=2.0cm of weights] (w_skip) {{Conv\\$1\times 1$}};
    
    % Sum
    \node (sum) [sum, right=0.8cm of ifft];
    \node[relu, right=0.8cm of sum] (act) {{GELU}};

    \begin{{scope}}[on background layer]
        \node[block, fit=(fft) (ifft) (w_skip) (act), label=above:Fourier Layer] (fno_layer) {{}};
    \end{{scope}}

    % Connections
    \draw[arrow] (lift) -- (fft);
    \draw[arrow] (fft) -- (weights);
    \draw[arrow] (weights) -- (ifft);
    \draw[arrow] (ifft) -- (sum);
    \draw[arrow] (sum) -- (act);
    
    % Skip path
    \draw[arrow] (lift) -- ++(0.5,0) |- (w_skip.west);
    \draw[arrow] (w_skip.east) -| (sum.north);

    % Projection
    \node[conv, right=2.0cm of act] (proj1) {{Proj 1\\$1\times 1$}};
    \node[conv, right=0.8cm of proj1] (proj2) {{Proj 2\\$1\times 1$}};
    \node[input, right=0.8cm of proj2] (out) {{Output}};

    \draw[arrow] (act) -- (proj1);
    \draw[arrow] (proj1) -- (proj2);
    \draw[arrow] (proj2) -- (out);

\end{{tikzpicture}}
\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_segformer_2d(out_tex):
    header = get_nntikz_header()
    tex = rf"""{header}

    \node[input] (in) {{Input}};
    
    % Overlap Patch Embed
    \node[conv, right=of in] (pe) {{Overlap\\PatchEmbed\\($7\times 7, s=4$)}};
    \draw[arrow] (in) -- (pe);

    % Mix Transformer Block
    \node[layer, right=2.0cm of pe] (ln1) {{Layer\\Norm}};
    
    % Attention Branch
    \node[layer, right=0.8cm of ln1, fill=green!10] (attn) {{Efficient\\Self-Attn}};
    \node (sum1) [sum, right=0.8cm of attn];
    
    \node[layer, right=1.2cm of sum1] (ln2) {{Layer\\Norm}};
    \node[layer, right=0.8cm of ln2, fill=orange!10] (ffn) {{Mix-FFN\\($3\times 3$ DW)}};
    \node (sum2) [sum, right=0.8cm of ffn];

    \begin{{scope}}[on background layer]
        \node[block, fit=(ln1) (sum2), label=above:MixTransformer Block (Stage 1)] (block1) {{}};
    \end{{scope}}

    % Connections
    \draw[arrow] (pe) -- (ln1);
    \draw[arrow] (ln1) -- (attn);
    \draw[arrow] (attn) -- (sum1);
    \draw[arrow] (sum1) -- (ln2);
    \draw[arrow] (ln2) -- (ffn);
    \draw[arrow] (ffn) -- (sum2);

    % Skips
    \draw[arrow] (pe) -- ++(0.8,0) |- ($(ln1.north) + (0, 0.8)$) -| (sum1.north);
    \draw[arrow] (sum1) -- ++(0.6,0) |- ($(ln2.north) + (0, 0.8)$) -| (sum2.north);

    % Stages 2-4
    \node[right=1.2cm of block1] (dots) {{\huge $\dots$}};
    \draw[arrow] (sum2) -- (dots);

    \node[block, right=1.2cm of dots] (stage4) {{Stage 4}};
    \draw[arrow] (dots) -- (stage4);

    % Head
    \node[layer, right=of stage4] (head) {{MLP\\Decoder}};
    \node[input, right=of head] (out) {{Output}};
    
    \draw[arrow] (stage4) -- (head);
    \draw[arrow] (head) -- (out);

\end{{tikzpicture}}
\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_swint_2d(out_tex):
    header = get_nntikz_header()
    tex = rf"""{header}

    \node[input] (in) {{Input}};
    
    % Patch Partition
    \node[layer, right=of in] (embed) {{Patch\\Partition}};
    \draw[arrow] (in) -- (embed);
    
    \node[layer, right=0.8cm of embed] (lin) {{Linear\\Embed}};
    \draw[arrow] (embed) -- (lin);

    % Swin Block
    \node[layer, right=2.0cm of lin] (ln1) {{LN}};
    \node[layer, right=0.8cm of ln1, fill=green!10] (wmsa) {{W-MSA\\(Shifted)}};
    \node (sum1) [sum, right=0.8cm of wmsa];
    
    \node[layer, right=1.2cm of sum1] (ln2) {{LN}};
    \node[layer, right=0.8cm of ln2, fill=yellow!10] (mlp) {{MLP}};
    \node (sum2) [sum, right=0.8cm of mlp];

    \begin{{scope}}[on background layer]
        \node[block, fit=(ln1) (sum2), label=above:Swin Transformer Block] (swin_block) {{}};
    \end{{scope}}

    % Connections
    \draw[arrow] (lin) -- (ln1);
    \draw[arrow] (ln1) -- (wmsa);
    \draw[arrow] (wmsa) -- (sum1);
    \draw[arrow] (sum1) -- (ln2);
    \draw[arrow] (ln2) -- (mlp);
    \draw[arrow] (mlp) -- (sum2);

    % Skips
    \draw[arrow] (lin) -- ++(0.8,0) |- ($(ln1.north) + (0, 0.8)$) -| (sum1.north);
    \draw[arrow] (sum1) -- ++(0.6,0) |- ($(ln2.north) + (0, 0.8)$) -| (sum2.north);

    % Patch Merging
    \node[layer, right=2.0cm of swin_block, fill=red!10] (merge) {{Patch\\Merging}};
    \draw[arrow] (sum2) -- (merge);

    % Stages
    \node[right=1.2cm of merge] (dots) {{\huge $\dots$}};
    \draw[arrow] (merge) -- (dots);
    
    \node[input, right=of dots] (out) {{Output}};
    \draw[arrow] (dots) -- (out);

\end{{tikzpicture}}
\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_swin_unet_2d(out_tex):
    header = get_nntikz_header()
    tex = rf"""{header}

    % Input
    \node[input] (in) {{Input}};

    % --- Encoder ---
    
    % Patch Embed
    \node[conv, right=of in] (pe) {{PatchEmbed\\($4\times 4$)}};
    \draw[arrow] (in) -- (pe);

    % Encoder Stage 1
    \node[block, right=1.5cm of pe, fill=green!10] (enc1) {{Swin Block\\$\times 2$}};
    \draw[arrow] (pe) -- (enc1);

    % Patch Merging 1 (Down)
    \node[layer, below=1.5cm of enc1, fill=red!10] (merge1) {{PatchMerging}};
    \draw[arrow] (enc1.south) -- (merge1.north);

    % Encoder Stage 2
    \node[block, below=1.5cm of merge1, fill=green!10] (enc2) {{Swin Block\\$\times 2$}};
    \draw[arrow] (merge1) -- (enc2);

    % Patch Merging 2
    \node[layer, below=1.5cm of enc2, fill=red!10] (merge2) {{PatchMerging}};
    \draw[arrow] (enc2) -- (merge2);

    % Bottleneck
    \node[block, below=1.5cm of merge2, fill=blue!10] (bot) {{Swin Block\\$\times 6$}};
    \draw[arrow] (merge2) -- (bot);

    % --- Decoder ---

    % Patch Expanding 2 (Up - align right)
    \node[layer, right=3.5cm of bot, fill=yellow!10] (expand2) {{PatchExpanding}};
    \draw[arrow] (bot) -- (expand2);

    % Decoder Stage 2
    \node[block, at={{(expand2 |- enc2)}}, fill=green!10] (dec2) {{Swin Block\\$\times 2$}};
    \draw[arrow] (expand2) -- (dec2);

    % Skip 2
    \draw[arrow, dashed] (enc2.east) -- node[above]{{Skip}} (dec2.west);

    % Patch Expanding 1
    \node[layer, at={{(dec2 |- merge1)}}, fill=yellow!10] (expand1) {{PatchExpanding}};
    \draw[arrow] (dec2) -- (expand1);

    % Decoder Stage 1
    \node[block, at={{(expand1 |- enc1)}}, fill=green!10] (dec1) {{Swin Block\\$\times 2$}};
    \draw[arrow] (expand1) -- (dec1);

    % Skip 1
    \draw[arrow, dashed] (enc1.east) -- node[above]{{Skip}} (dec1.west);

    % Output
    \node[layer, right=1.5cm of dec1, fill=yellow!10] (final_expand) {{PatchExpanding\\($\times 4$)}};
    \node[input, right=of final_expand] (out) {{Output}};
    
    \draw[arrow] (dec1) -- (final_expand);
    \draw[arrow] (final_expand) -- (out);

\end{{tikzpicture}}
\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_hybrid_2d(out_tex):
    header = get_nntikz_header()
    tex = rf"""{header}

    \node[input] (in) {{Input}};
    
    % Split point
    \coordinate[right=1.0cm of in] (split);
    \draw[arrow] (in) -- (split);

    % --- Top Branch: Attention ---
    \node[block, above right=1.5cm and 2.0cm of split, fill=green!5, align=center] (attn_branch) {{Attention Branch\\(Window Attn)}};
    \draw[arrow] (split) |- (attn_branch.west);

    % --- Middle Branch: FNO ---
    \node[block, right=2.0cm of split, fill=blue!5, align=center] (fno_branch) {{FNO Branch\\(FFT $\to$ Spec $\to$ IFFT)}};
    \draw[arrow] (split) -- (fno_branch.west);

    % --- Bottom Branch: UNet ---
    \node[block, below right=1.5cm and 2.0cm of split, fill=orange!5, align=center] (unet_branch) {{UNet Branch\\(Encoder-Decoder)}};
    \draw[arrow] (split) |- (unet_branch.west);

    % --- Fusion ---
    \node[circle, draw, right=2.0cm of fno_branch, minimum size=3em, fill=gray!10] (fusion) {{Fuse}};
    
    \draw[arrow] (attn_branch.east) -| (fusion.north);
    \draw[arrow] (fno_branch.east) -- (fusion.west);
    \draw[arrow] (unet_branch.east) -| (fusion.south);

    % Head
    \node[conv, right=1.5cm of fusion] (head) {{Conv Head}};
    \node[input, right=of head] (out) {{Output}};

    \draw[arrow] (fusion) -- (head);
    \draw[arrow] (head) -- (out);

\end{{tikzpicture}}
\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_liif_2d(out_tex):
    header = get_nntikz_header()
    tex = rf"""{header}

    \node[input] (in) {{Input}};
    
    % Encoder
    \node[block, right=1.5cm of in, fill=orange!10] (encoder) {{Encoder\\(EDSR/CNN)}};
    \draw[arrow] (in) -- (encoder);

    % Feature Grid
    \node[layer, right=1.5cm of encoder, fill=blue!10] (feat) {{Feature Map\\(Grid)}};
    \draw[arrow] (encoder) -- (feat);

    % Coord Input
    \node[input, below=2.0cm of in, fill=yellow!10] (coord) {{Coordinates\\$(x_q, y_q)$}};
    
    % Sampling (Query)
    \node[block, at={{(feat |- coord)}}, fill=gray!5] (sample) {{Query/Sample\\(Local Ensemble)}};
    \draw[arrow] (coord) -- (sample);
    
    % Feature to Sample
    \draw[arrow] (feat.south) -- node[right]{{Unfold}} (sample.north);

    % MLP
    \node[block, right=2.0cm of sample, fill=green!10] (mlp) {{MLP\\($f_\theta$)}};
    \draw[arrow] (sample) -- (mlp);

    % Output
    \node[input, right=of mlp] (out) {{RGB Value}};
    \draw[arrow] (mlp) -- (out);

\end{{tikzpicture}}
\end{{document}}
"""
    out_tex.write(tex)

def gen_tikz_mlp_mixer_2d(out_tex):
    """
    Generate TikZ code for MLP-Mixer (Dense Prediction).
    Structure: PatchEmbed -> Mixer Block x N -> PatchRestore
    """
    tikz = r"""
\documentclass[tikz, border=10pt]{standalone}
\usepackage{tikz}
\usepackage{graphicx}
\usetikzlibrary{positioning, shadows, calc, shapes}

\begin{document}
\section{MLP-Mixer Architecture}
\begin{tikzpicture}[
    node distance=1.5cm,
    block/.style={draw, rectangle, minimum height=1.2cm, minimum width=2.5cm, align=center, fill=white, drop shadow},
    sqblock/.style={draw, rectangle, minimum size=1.2cm, align=center, fill=white, drop shadow},
    arrow/.style={->, >=stealth, thick}
]
    % Input
    \node (input) {\includegraphics[width=1.5cm]{example-image-a}};
    \node[below=0.1cm of input] {Input ($H \times W$)};

    % Patch Embed
    \node[block, right=2.0cm of input] (patchembed) {Patch Embed\\(Conv $P \times P$)};
    
    % Mixer Block 1
    \node[block, right=2.0cm of patchembed] (mixer1) {Mixer Block 1\\Token Mix $\to$ Channel Mix};
    
    % Mixer Block 2
    \node[block, right=2.0cm of mixer1] (mixer2) {Mixer Block 2\\...};
    
    % Patch Restore
    \node[block, right=2.0cm of mixer2] (restore) {Patch Restore\\(Linear + Reshape)};
    
    % Output
    \node[right=2.0cm of restore] (output) {\includegraphics[width=1.5cm]{example-image-b}};
    \node[below=0.1cm of output] {Output ($H \times W$)};

    % Connections
    \draw[arrow] (input) -- node[above] {Tokens} (patchembed);
    \draw[arrow] (patchembed) -- (mixer1);
    \draw[arrow] (mixer1) -- (mixer2);
    \draw[arrow] (mixer2) -- (restore);
    \draw[arrow] (restore) -- (output);
    
    % Internal detail of Mixer Block (Optional, simplified here)
    \node[below=1.0cm of mixer1, font=\small, align=center] {Token Mix: MLP on $N$\\Channel Mix: MLP on $C$};

\end{tikzpicture}
\end{document}
"""
    out_tex.write(tikz)

def gen_tikz_deeponet_2d(out_tex):
    """
    Generate TikZ code for DeepONet.
    Structure: Branch (Image) + Trunk (Coords) -> Dot Product
    """
    tikz = r"""
\documentclass[tikz, border=10pt]{standalone}
\usepackage{tikz}
\usepackage{graphicx}
\usepackage{amsmath}
\usetikzlibrary{positioning, shadows, calc, shapes}

\newcommand{\bigcdot}{\cdot} % Fallback

\begin{document}
\section{DeepONet Architecture}
\begin{tikzpicture}[
    node distance=1.5cm,
    block/.style={draw, rectangle, minimum height=1.0cm, minimum width=2.0cm, align=center, fill=white, drop shadow},
    op/.style={draw, circle, minimum size=0.8cm, fill=white, drop shadow},
    arrow/.style={->, >=stealth, thick}
]
    % Branch Net (Top)
    \node (input) {\includegraphics[width=1.5cm]{example-image-a}};
    \node[left=0.1cm of input] {Input $u$};
    
    \node[block, right=1.5cm of input] (branch1) {Conv Layers};
    \node[block, right=1.0cm of branch1] (branch2) {Global Pool};
    \node[block, right=1.0cm of branch2] (branch_out) {Branch Output\\$b_k$};

    % Trunk Net (Bottom)
    \node[below=3.0cm of input] (coords) {Coords $(x,y)$};
    
    \node[block, right=1.5cm of coords] (trunk1) {Fourier Feat};
    \node[block, right=1.0cm of trunk1] (trunk2) {MLP Layers};
    \node[block, right=1.0cm of trunk2] (trunk_out) {Trunk Output\\$t_k$};
    
    % Combine
    \node[op, at={($(branch_out)!0.5!(trunk_out)$)}, right=2.0cm] (dot) {$\bigcdot$};
    \node[above=0.1cm of dot] {Dot Product};

    % Output
    \node[right=1.5cm of dot] (output) {\includegraphics[width=1.5cm]{example-image-b}};
    \node[right=0.1cm of output] {Output $G(u)(y)$};

    % Connections Branch
    \draw[arrow] (input) -- (branch1);
    \draw[arrow] (branch1) -- (branch2);
    \draw[arrow] (branch2) -- (branch_out);
    \draw[arrow] (branch_out) -| (dot);

    % Connections Trunk
    \draw[arrow] (coords) -- (trunk1);
    \draw[arrow] (trunk1) -- (trunk2);
    \draw[arrow] (trunk2) -- (trunk_out);
    \draw[arrow] (trunk_out) -| (dot);

    % Connection Out
    \draw[arrow] (dot) -- (output);

\end{tikzpicture}
\end{document}
"""
    out_tex.write(tikz)

def gen_tikz_ufno_2d(out_tex):
    """
    Generate TikZ code for U-FNO (U-Net with FNO Bottleneck).
    """
    tikz = r"""
\documentclass[tikz, border=10pt]{standalone}
\usepackage{tikz}
\usepackage{graphicx}
\usetikzlibrary{positioning, shadows, calc, shapes}

\begin{document}
\section{U-FNO Architecture}
\begin{tikzpicture}[
    node distance=1.2cm,
    block/.style={draw, rectangle, minimum height=1.0cm, minimum width=1.5cm, align=center, fill=white, drop shadow},
    fno/.style={draw, rectangle, minimum height=1.2cm, minimum width=3.0cm, align=center, fill=blue!10, drop shadow},
    arrow/.style={->, >=stealth, thick},
    skip/.style={->, >=stealth, dashed}
]
    % Encoder
    \node (input) {\includegraphics[width=1.5cm]{example-image-a}};
    \node[below=0.1cm of input] {Input};

    \node[block, right=1.5cm of input] (enc1) {Enc 1\\(Conv)};
    \node[block, right=1.0cm of enc1] (enc2) {Enc 2\\(Down)};
    \node[block, right=1.0cm of enc2] (enc3) {Enc 3\\(Down)};

    % Bottleneck (FNO)
    \node[fno, below=2.0cm of enc2] (bottleneck) {FNO Bottleneck\\(Spectral Conv + FFT)};

    % Decoder
    \node[block, below=2.0cm of enc3] (dec3) {Dec 3\\(Up)};
    \node[block, right=1.0cm of dec3] (dec2) {Dec 2\\(Up)};
    \node[block, right=1.0cm of dec2] (dec1) {Dec 1\\(Up)};
    
    \node[right=1.5cm of dec1] (output) {\includegraphics[width=1.5cm]{example-image-b}};
    \node[below=0.1cm of output] {Output};

    % Encoder Flow
    \draw[arrow] (input) -- (enc1);
    \draw[arrow] (enc1) -- (enc2);
    \draw[arrow] (enc2) -- (enc3);
    
    % To Bottleneck
    \draw[arrow] (enc3) |- (bottleneck);
    
    % To Decoder
    \draw[arrow] (bottleneck) -| (dec3);
    
    % Decoder Flow
    \draw[arrow] (dec3) -- (dec2);
    \draw[arrow] (dec2) -- (dec1);
    \draw[arrow] (dec1) -- (output);

    % Skip Connections (U-Net style)
    % Ideally horizontal if layout permits, here we use simple curves or straight lines
    % We adjust positions to make them look like U-Net if we used a standard U-layout, 
    % but here we used a wrapped layout. Let's try to align Decoders under Encoders for better skips.
    
    % Re-positioning for U-shape alignment
    % Let's use coordinate extraction to align Dec under Enc
    % But we already placed them. Let's rely on relative placement above.
    % Actually, let's force alignment:
    % enc1 -- enc2 -- enc3
    % dec1 -- dec2 -- dec3
    % Skip: enc3 -> dec3 (if matched), enc2 -> dec2...
    
    % Let's redraw lines for skips
    % Note: In this specific layout (Enc top row, Dec bottom row), skips are vertical
    \draw[skip] (enc3) -- node[right, font=\tiny] {Skip} (dec3);
    \draw[skip] (enc2) -- (dec2);
    \draw[skip] (enc1) -- (dec1);

\end{tikzpicture}
\end{document}
"""
    out_tex.write(tikz)

def process_model(model_name):
    print(f"Generating 2D TikZ for {model_name}...")
    
    base_dir = os.path.dirname(__file__)
    export_dir = os.path.join(base_dir, 'build_export', '2d', model_name)
    os.makedirs(export_dir, exist_ok=True)
    
    tex_path = os.path.join(export_dir, f"fig_{model_name.lower()}_2d.tex")
    
    with open(tex_path, 'w') as f:
        if model_name == 'EDSR':
            gen_tikz_edsr_2d(f)
        elif model_name == 'UNet':
            gen_tikz_unet_2d(f)
        elif model_name == 'FNO2d':
            gen_tikz_fno2d_2d(f)
        elif model_name == 'SegFormer':
            gen_tikz_segformer_2d(f)
        elif model_name == 'SwinT':
            gen_tikz_swint_2d(f)
        elif model_name == 'SwinUNet':
            gen_tikz_swin_unet_2d(f)
        elif model_name == 'Hybrid':
            gen_tikz_hybrid_2d(f)
        elif model_name == 'LIIF':
            gen_tikz_liif_2d(f)
        elif model_name == 'MLP-Mixer':
            gen_tikz_mlp_mixer_2d(f)
        elif model_name == 'DeepONet':
            gen_tikz_deeponet_2d(f)
        elif model_name == 'U-FNO':
            gen_tikz_ufno_2d(f)
        else:
            print(f"Unknown model {model_name}")
            return

    # Compile
    cmd = ['conda', 'run', '-n', 'latex', 'tectonic', tex_path]
    print(f"Compiling: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"Success: {model_name} 2D PDF generated.")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {model_name}: {e}")

def main():
    models = ["EDSR", "UNet", "FNO2d", "SegFormer", "SwinT", "SwinUNet", "Hybrid", "LIIF", "MLP-Mixer", "DeepONet", "U-FNO"]
    
    for m in models:
        process_model(m)

if __name__ == "__main__":
    main()
