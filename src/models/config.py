class Cfg_SWM:
    # data
    train_csv: list
    dev_csv: list
    label_json: str

    # model
    text_model = "distilbert-base-uncased"
    freeze_text = False
    use_wavlm = True
    wavlm_in_dim = 1024 
    wavlm_out_dim = 64
    z_dim = 88

    tom_hidden = 128
    sa_hidden = 256
    wma_hidden = 256
    prag_hidden = 128

    num_emotions = 7  
    num_sa = 24
    num_wma = 30
    num_prag = 14

    tom_teacher_force_p = 0.3
    wma_teacher_force_p = 0.3
    sa_teacher_force_p = 0.3

    # causal edge
    sa_use_wma = True ##
    sa_use_tom = True ##

    # fuison
    pre_fusion = True
    sa_fusion = True
    prag_fusion = True
    
    fusion_type = 'gated'  # 'concat', 'attention', 'gated', 'transformer'
    fusion_hidden = 256
    fusion_layers = 2

    # temporal
    use_text_temporal = True
    use_wavlm_temporal = True
    use_temporal_tom = True

    use_tom_temporal_attn = True
    use_wma_temporal_attn = True

    sa_use_residual = True
    prag_use_residual = True

    lambda_wma = 1.0
    lambda_tom = 1.0
    lambda_sa = 1.0
    lambda_prag = 1.0

    # train
    device = "cuda:6"
    epochs = 20
    batch_size = 32
    lr_text = 1e-5
    lr_head = 1e-3
    weight_decay = 0.01
    lambda_sa = 1.0
    max_grad_norm = 1.0
    amp = False
    log_every = 50
    out_ckpt = "best_full.pt"

    def get(self, key, default=None):
        return getattr(self, key, default)