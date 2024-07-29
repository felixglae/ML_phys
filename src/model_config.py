from transformers.models.segformer import SegformerConfig


class SegFormerConfigFactory:
    _NUM_ENCODER_BLOCKS = 4
    _SR_RATIOS = [8, 4, 2, 1]
    _PATCH_SIZES = [7, 3, 3, 3]
    _STRIDES = [4, 2, 2, 2]
    _NUM_ATTENTION_HEADS = [1, 2, 5, 8]
    _MLP_RATIOS = [4, 4, 4, 4]
    _HIDDEN_ACT = 'gelu'
    _HIDDEN_DROPOUT_PROB = 0.
    _ATTENTION_PROBS_DROPOUT_PROB = 0.
    _CLASSIFIER_DROPOUT_PROB = .1
    _INITIALIZER_RANGE = .02
    _DROP_PATH_RATE = .1
    _LAYER_NORM_EPS = 1e-6
    _SEMANTIC_LOSS_IGNORE_INDEX = 255

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
    ) -> None:
        self.num_channels = num_channels
        self.num_classes = num_classes

    @property
    def b0_config(self) -> SegformerConfig:
        depths = [2, 2, 2, 2]
        hidden_sizes = [32, 64, 160, 256]
        decoder_hidden_size = 256

        return SegformerConfig(
            num_channels=self.num_channels,
            num_labels=self.num_classes,
            num_encoder_blocks=self._NUM_ENCODER_BLOCKS,
            depths=depths,
            sr_ratios=self._SR_RATIOS,
            hidden_sizes=hidden_sizes,
            patch_sizes=self._PATCH_SIZES,
            strides=self._STRIDES,
            num_attention_heads=self._NUM_ATTENTION_HEADS,
            mlp_ratios=self._MLP_RATIOS,
            hidden_act=self._HIDDEN_ACT,
            hidden_dropout_prob=self._HIDDEN_DROPOUT_PROB,
            attention_probs_dropout_prob=self._ATTENTION_PROBS_DROPOUT_PROB,
            classifier_dropout_prob=self._CLASSIFIER_DROPOUT_PROB,
            initializer_range=self._INITIALIZER_RANGE,
            drop_path_rate=self._DROP_PATH_RATE,
            layer_norm_eps=self._LAYER_NORM_EPS,
            decoder_hidden_size=decoder_hidden_size,
            semantic_loss_ignore_index=self._SEMANTIC_LOSS_IGNORE_INDEX,
        )

    @property
    def b1_config(self) -> SegformerConfig:
        depths = [2, 2, 2, 2]
        hidden_sizes = [64, 128, 320, 512]
        decoder_hidden_size = 256

        return SegformerConfig(
            num_channels=self.num_channels,
            num_labels=self.num_classes,
            num_encoder_blocks=self._NUM_ENCODER_BLOCKS,
            depths=depths,
            sr_ratios=self._SR_RATIOS,
            hidden_sizes=hidden_sizes,
            patch_sizes=self._PATCH_SIZES,
            strides=self._STRIDES,
            num_attention_heads=self._NUM_ATTENTION_HEADS,
            mlp_ratios=self._MLP_RATIOS,
            hidden_act=self._HIDDEN_ACT,
            hidden_dropout_prob=self._HIDDEN_DROPOUT_PROB,
            attention_probs_dropout_prob=self._ATTENTION_PROBS_DROPOUT_PROB,
            classifier_dropout_prob=self._CLASSIFIER_DROPOUT_PROB,
            initializer_range=self._INITIALIZER_RANGE,
            drop_path_rate=self._DROP_PATH_RATE,
            layer_norm_eps=self._LAYER_NORM_EPS,
            decoder_hidden_size=decoder_hidden_size,
            semantic_loss_ignore_index=self._SEMANTIC_LOSS_IGNORE_INDEX,
        )

    @property
    def b2_config(self) -> SegformerConfig:
        depths = [3, 4, 6, 3]
        hidden_sizes = [64, 128, 320, 512]
        decoder_hidden_size = 768

        return SegformerConfig(
            num_channels=self.num_channels,
            num_labels=self.num_classes,
            num_encoder_blocks=self._NUM_ENCODER_BLOCKS,
            depths=depths,
            sr_ratios=self._SR_RATIOS,
            hidden_sizes=hidden_sizes,
            patch_sizes=self._PATCH_SIZES,
            strides=self._STRIDES,
            num_attention_heads=self._NUM_ATTENTION_HEADS,
            mlp_ratios=self._MLP_RATIOS,
            hidden_act=self._HIDDEN_ACT,
            hidden_dropout_prob=self._HIDDEN_DROPOUT_PROB,
            attention_probs_dropout_prob=self._ATTENTION_PROBS_DROPOUT_PROB,
            classifier_dropout_prob=self._CLASSIFIER_DROPOUT_PROB,
            initializer_range=self._INITIALIZER_RANGE,
            drop_path_rate=self._DROP_PATH_RATE,
            layer_norm_eps=self._LAYER_NORM_EPS,
            decoder_hidden_size=decoder_hidden_size,
            semantic_loss_ignore_index=self._SEMANTIC_LOSS_IGNORE_INDEX,
        )

    @property
    def b3_config(self) -> SegformerConfig:
        depths = [3, 4, 18, 3]
        hidden_sizes = [64, 128, 320, 512]
        decoder_hidden_size = 768

        return SegformerConfig(
            num_channels=self.num_channels,
            num_labels=self.num_classes,
            num_encoder_blocks=self._NUM_ENCODER_BLOCKS,
            depths=depths,
            sr_ratios=self._SR_RATIOS,
            hidden_sizes=hidden_sizes,
            patch_sizes=self._PATCH_SIZES,
            strides=self._STRIDES,
            num_attention_heads=self._NUM_ATTENTION_HEADS,
            mlp_ratios=self._MLP_RATIOS,
            hidden_act=self._HIDDEN_ACT,
            hidden_dropout_prob=self._HIDDEN_DROPOUT_PROB,
            attention_probs_dropout_prob=self._ATTENTION_PROBS_DROPOUT_PROB,
            classifier_dropout_prob=self._CLASSIFIER_DROPOUT_PROB,
            initializer_range=self._INITIALIZER_RANGE,
            drop_path_rate=self._DROP_PATH_RATE,
            layer_norm_eps=self._LAYER_NORM_EPS,
            decoder_hidden_size=decoder_hidden_size,
            semantic_loss_ignore_index=self._SEMANTIC_LOSS_IGNORE_INDEX,
        )

    @property
    def b4_config(self) -> SegformerConfig:
        depths = [3, 8, 27, 3]
        hidden_sizes = [64, 128, 320, 512]
        decoder_hidden_size = 768

        return SegformerConfig(
            num_channels=self.num_channels,
            num_labels=self.num_classes,
            num_encoder_blocks=self._NUM_ENCODER_BLOCKS,
            depths=depths,
            sr_ratios=self._SR_RATIOS,
            hidden_sizes=hidden_sizes,
            patch_sizes=self._PATCH_SIZES,
            strides=self._STRIDES,
            num_attention_heads=self._NUM_ATTENTION_HEADS,
            mlp_ratios=self._MLP_RATIOS,
            hidden_act=self._HIDDEN_ACT,
            hidden_dropout_prob=self._HIDDEN_DROPOUT_PROB,
            attention_probs_dropout_prob=self._ATTENTION_PROBS_DROPOUT_PROB,
            classifier_dropout_prob=self._CLASSIFIER_DROPOUT_PROB,
            initializer_range=self._INITIALIZER_RANGE,
            drop_path_rate=self._DROP_PATH_RATE,
            layer_norm_eps=self._LAYER_NORM_EPS,
            decoder_hidden_size=decoder_hidden_size,
            semantic_loss_ignore_index=self._SEMANTIC_LOSS_IGNORE_INDEX,
        )

    @property
    def b5_config(self) -> SegformerConfig:
        depths = [3, 6, 40, 3]
        hidden_sizes = [64, 128, 320, 512]
        decoder_hidden_size = 768

        return SegformerConfig(
            num_channels=self.num_channels,
            num_labels=self.num_classes,
            num_encoder_blocks=self._NUM_ENCODER_BLOCKS,
            depths=depths,
            sr_ratios=self._SR_RATIOS,
            hidden_sizes=hidden_sizes,
            patch_sizes=self._PATCH_SIZES,
            strides=self._STRIDES,
            num_attention_heads=self._NUM_ATTENTION_HEADS,
            mlp_ratios=self._MLP_RATIOS,
            hidden_act=self._HIDDEN_ACT,
            hidden_dropout_prob=self._HIDDEN_DROPOUT_PROB,
            attention_probs_dropout_prob=self._ATTENTION_PROBS_DROPOUT_PROB,
            classifier_dropout_prob=self._CLASSIFIER_DROPOUT_PROB,
            initializer_range=self._INITIALIZER_RANGE,
            drop_path_rate=self._DROP_PATH_RATE,
            layer_norm_eps=self._LAYER_NORM_EPS,
            decoder_hidden_size=decoder_hidden_size,
            semantic_loss_ignore_index=self._SEMANTIC_LOSS_IGNORE_INDEX,
        )
