# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import fields
from mcore_bridge import ModelConfig
from mcore_bridge import get_mcore_model as _get_mcore_model
from mcore_bridge import hf_to_mcore_config
from transformers.utils import is_torch_npu_available

from swift.utils import get_logger

logger = get_logger()


def _check_attention_backend(args, config):
    """Validate attention backend compatibility with configuration."""
    attention_backend = config.attention_backend.name
    if attention_backend == 'flash' and config.softmax_type == 'learnable':
        raise ValueError(f'Attention backend "{attention_backend}" does not support learnable softmax_type.')


def _check_te_rmsnorm_compat(args, config):
    """
    Test whether TransformerEngine's RMSNorm kernel can handle the model's hidden_size.

    When padding_free=True, TELayerNormColumnParallelLinear may fall back from its
    fused LayerNorm+Linear kernel to a standalone RMSNorm kernel (e.g., for custom TE
    builds that lack fused-kernel THD/packed-sequence support).  The standalone
    "general" RMSNorm fallback kernel launches with blockDim.x = hidden_size, which
    exceeds the 1024-thread GPU limit for any model with hidden_size > 1024, producing:

        CUDA Error: invalid configuration argument (launch_rmsnorm_fwd_general_)

    We probe for this at startup by running a dry forward through te.RMSNorm with the
    actual hidden_size.  If it fails we automatically disable padding_free so the
    fused kernel (which has its own per-size specialisations) is used instead, and
    emit a clear warning pointing at the root cause.
    """
    hidden_size = getattr(config, 'hidden_size', None)
    if hidden_size is None or hidden_size <= 1024:
        # The general kernel launches with min(hidden_size, 1024) threads on most
        # TE versions, so sizes ≤ 1024 are always safe.
        return

    try:
        import torch
        import transformer_engine.pytorch as te

        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        params_dtype = getattr(config, 'params_dtype', None) or torch.bfloat16

        norm = te.RMSNorm(hidden_size).to(device=device, dtype=params_dtype)
        test_input = torch.zeros(1, hidden_size, device=device, dtype=params_dtype)
        try:
            norm(test_input)
        except RuntimeError as cuda_err:
            err_msg = str(cuda_err)
            if 'invalid configuration argument' in err_msg or 'CUDA Error' in err_msg:
                logger.warning(
                    f'TransformerEngine RMSNorm kernel failed for hidden_size={hidden_size} '
                    f'with "CUDA Error: invalid configuration argument". '
                    f'This typically means the TransformerEngine build is missing '
                    f'specialized CUDA kernels for this hidden_size on your GPU architecture '
                    f'(e.g. a custom or source-built TE that was not compiled with full kernel support). '
                    f'Automatically setting args.padding_free=False so the fused '
                    f'TELayerNormColumnParallelLinear kernel is used instead. '
                    f'To resolve permanently, rebuild TransformerEngine with proper CUDA support '
                    f'or use an official TransformerEngine release.')
                args.padding_free = False
        finally:
            del norm, test_input
            torch.cuda.empty_cache()
    except Exception:
        pass  # Never block training startup on the compatibility probe itself


def _check_padding_free(args, config):
    """Validate and adjust padding_free setting based on configuration constraints."""
    if not args.padding_free:
        return

    attention_backend = config.attention_backend.name
    message = None

    if config.experimental_attention_variant == 'dsa':
        message = 'DSA is not supported in padding-free mode'
    elif attention_backend == 'unfused':
        message = f'Attention backend "{attention_backend}" is not supported in padding-free mode'

    if message:
        logger.warning(f'{message}. Setting args.padding_free to False.')
        args.padding_free = False
        return

    # Verify TE's RMSNorm kernel can handle the model's hidden_size in packed-sequence
    # (THD) mode.  This catches custom TE builds where the fused path falls back to
    # the standalone general kernel which fails for hidden_size > 1024.
    _check_te_rmsnorm_compat(args, config)


_SWIFT_ONLY_ARGS = frozenset({'enable_loop', 'loop_method', 'loop_times'})


def get_mcore_model_config(args, hf_config):
    kwargs = hf_to_mcore_config(hf_config)
    kwargs['mcore_model_type'] = args.megatron_model_meta.model_type
    kwargs['hf_config'] = hf_config
    for f in fields(ModelConfig):
        key, value = f.name, getattr(args, f.name, None)
        if key in _SWIFT_ONLY_ARGS:
            continue
        if value is None or isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        kwargs[key] = value

    if args.task_type == 'seq_cls':
        args.problem_type = args.problem_type or getattr(hf_config, 'problem_type', None)
        logger.info(f'args.problem_type: {args.problem_type}')

    kwargs['params_dtype'] = args.torch_dtype
    kwargs['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kwargs['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers
    kwargs['fp8_param'] = args.fp8_param_gather
    swiglu = kwargs.get('swiglu', True)
    add_bias_linear = kwargs.get('add_bias_linear', False)
    num_moe_experts = kwargs.get('num_moe_experts', None)
    position_embedding_type = kwargs.get('position_embedding_type', 'rope')
    if position_embedding_type != 'rope':
        kwargs['apply_rope_fusion'] = False
    if not swiglu and not add_bias_linear:
        kwargs['bias_activation_fusion'] = False
    if add_bias_linear and num_moe_experts and args.moe_grouped_gemm:
        kwargs['bias_dropout_fusion'] = False
    if num_moe_experts is None:
        kwargs['expert_model_parallel_size'] = 1
        kwargs['expert_tensor_parallel_size'] = 1

    if args.router_replay_mode != 'disabled':
        kwargs['moe_enable_routing_replay'] = True
    config = ModelConfig(**kwargs)
    if is_torch_npu_available() and getattr(args, 'attention_backend', 'flash') != 'local':
        setattr(config, 'use_flash_attn', True)
    _check_attention_backend(args, config)
    _check_padding_free(args, config)
    return config


def get_mcore_model(args, hf_config):
    config = get_mcore_model_config(args, hf_config)
    models = _get_mcore_model(config)

    return models
