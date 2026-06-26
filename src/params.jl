# Tunable search parameters — all weights use ÷ 1024 as denominator

# Aspiration windows
const asp_init_window  = 35
const asp_inf_thresh   = 200

# LMR formula: R = 1 + log(d) * log(i) / (lmr_div / 1000)
const lmr_div = 3000

# Reverse futility pruning
const rfp_mult             = (140, 145, 155, 160, 165, 175, 200)  # per depth 1–7
const rfp_improving_offset = 90

# Razoring
const razor_margin_1 = 350
const razor_margin_2 = 1200

# SEE pruning coefficients
const see_capture_coeff = 45
const see_quiet_coeff   = 80

# Singular extension thresholds
const sing_double_thresh = 20
const sing_triple_thresh = 40

# Mini-probcut
const mini_pc_margin = 500

# LMR continuation-history reduction threshold
const lmr_ch_thresh = 1000

# Move ordering weights (÷ 1024)
const cont_hist_w  = 1024   # cont hist 1
const cont_hist2_w = 1024   # cont hist 2
const pawn_hist_w  =  435   # pawn hist
const cap_hist_w   =   56   # capture hist 

# Correction history weights (÷ 1024, then ÷ Δ)
const corr_pawn_w    = 1024
const corr_minor_w   = 1024
const corr_major_w_w =  512
const corr_major_b_w =  512
