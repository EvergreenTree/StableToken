from transformers import WhisperConfig


class WhisperVQConfig(WhisperConfig):
    def __init__(self,
                 pooling_kernel_size=None,
                 pooling_type="max",
                 pooling_position=16,
                 quantize_encoder_only=False,
                 quantize_vocab_size=None,
                 quantize_position=16,
                 quantize_commit_coefficient=0.25,
                 num_voters=3,
                 num_clean_input=2,
                 use_codebook_ce_loss=False,
                 use_commit_loss=True,
                 consensus_loss_weight=0.25,
                 codebook_entropy_loss_weight=1.0,
                 batch_maximization_weight=1.0,
                 sample_minimization_weight=1.0,
                 layernorm_after_quantize=False,
                 use_projection_bias=True,
                 encoder_causal_convolution=False,
                 **kwargs):
        """
        Initialize WhisperVQ configuration.

        Args:
            pooling_kernel_size (int, optional): Kernel size for pooling operation.
                Used for temporal downsampling of features.
            pooling_type (str): Type of pooling operation. Options: "max", "avg".
                Controls how temporal features are aggregated.
            pooling_position (int): Transformer layer index where pooling is applied.
                Determines at which depth temporal compression occurs.
            quantize_encoder_only (bool): If True, only quantizes encoder outputs.
                If False, quantizes the full model pipeline.
            quantize_vocab_size (int, optional): Size of the quantization codebook.
                Determines the number of discrete tokens available.
            quantize_position (int): Transformer layer index where quantization is applied.
                Controls where in the network discretization occurs.
            quantize_commit_coefficient (float): Coefficient for commitment loss.
                Balances between reconstruction and commitment to codebook entries.
            num_voters (int): Number of voting heads in Voting-LFQ mechanism.
                More voters can improve robustness but increase computation.
            num_clean_input (int): Number of clean reference inputs for noise robustness.
                Used in training to maintain semantic consistency under noise.
            use_codebook_ce_loss (bool): Whether to apply cross-entropy loss on codebook.
                Additional regularization for better codebook utilization.
            use_commit_loss (bool): Whether to apply commitment loss in quantization.
                Encourages encoder outputs to commit to codebook entries.
            batch_maximization_weight (float): Weight for batch-level diversity loss.
                Encourages diverse token usage across batch samples.
            sample_minimization_weight (float): Weight for sample-level consistency loss.
                Encourages consistent token usage within each sample.
            layernorm_after_quantize (bool): Whether to normalize after quantization.
                Can help stabilize training with discrete representations.
            use_projection_bias (bool): Whether to use bias terms in projection layers.
                Affects the linear transformations in the quantization module.
            encoder_causal_convolution (bool): Whether encoder convolutions should be
                causal. StableToken inference uses Whisper-style non-causal chunks by
                default.
            **kwargs: Additional arguments passed to parent WhisperConfig.
        """
        # Pooling configuration - controls temporal feature compression
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position
        self.quantize_vocab_size = quantize_vocab_size
        self.quantize_position = quantize_position
        self.quantize_commit_coefficient = quantize_commit_coefficient
        self.quantize_encoder_only = quantize_encoder_only
        self.num_voters = num_voters
        self.num_clean_input = num_clean_input
        self.use_codebook_ce_loss = use_codebook_ce_loss
        self.use_commit_loss = use_commit_loss
        self.consensus_loss_weight = consensus_loss_weight
        self.codebook_entropy_loss_weight = codebook_entropy_loss_weight
        self.batch_maximization_weight = batch_maximization_weight
        self.sample_minimization_weight = sample_minimization_weight
        self.layernorm_after_quantize = layernorm_after_quantize
        self.use_projection_bias = use_projection_bias
        self.encoder_causal_convolution = encoder_causal_convolution
        super().__init__(**kwargs)
