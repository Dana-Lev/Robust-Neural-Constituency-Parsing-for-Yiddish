# -*- coding: utf-8 -*-
r"""
Approach 2: MAD-X-style Yiddish language adapters for the SuPar constituency parser.

WHERE THE ADAPTERS GO
---------------------
SuPar builds the encoder in ``supar/models/model.py``::

    self.encoder = TransformerEmbedding(model=bert, n_layers=n_bert_layers,
                                        pooling=bert_pooling, pad_index=pad_index,
                                        dropout=mix_dropout, requires_grad=True)

and ``TransformerEmbedding.__init__`` does::

    self.bert = AutoModel.from_pretrained(model, ...)
    self.bert = self.bert.requires_grad_(requires_grad)   # <-- True, unconditionally

So: ``parser.model.encoder`` is the SuPar wrapper (scalar mix + projection) and
``parser.model.encoder.bert`` is the plain HF ``XLMRobertaModel``. Note that
SuPar hard-codes ``requires_grad=True`` regardless of the ``finetune`` flag --
this is why the original ``train_parser.py`` needs a runtime freeze patch, and
why we freeze explicitly here rather than trusting any config value.

TWO ADAPTER FAMILIES
--------------------
``lora``
    ``peft.inject_adapter_in_model`` with a ``LoraConfig`` targeting the
    self-attention query/value projections. We use ``inject_adapter_in_model``
    rather than ``get_peft_model`` on purpose: it mutates the module tree in
    place and returns a plain ``XLMRobertaModel``, so SuPar's
    ``TransformerEmbedding.forward`` -- which calls ``self.bert(...)[-1]`` and
    indexes into the output tuple -- keeps working untouched, and no ``PeftModel``
    wrapper interferes with SuPar's ``state_dict`` round-trip.

``bottleneck``
    A faithful Pfeiffer-style bottleneck adapter (MAD-X; Pfeiffer et al. 2020):
    ``h + W_up(GELU(W_down(h)))`` applied after each transformer layer's output
    LayerNorm. ``peft`` has no bottleneck-adapter tuner, so this is implemented
    directly with forward hooks. Hooks are deliberate: they add **zero** new
    entries under the XLM-R submodule names, so the frozen backbone's
    ``state_dict`` keys stay byte-identical to the baseline checkpoint. The
    adapter parameters are registered separately on the ``TransformerEmbedding``
    as ``encoder.language_adapters.*``, so SuPar still saves and restores them.

    ``W_up`` is near-zero initialised, so at step 0 the adapted model is exactly
    the frozen baseline -- the adapters cannot hurt the starting point.

FREEZING
--------
:func:`freeze_backbone` freezes every backbone parameter, including the
``250002 x 768 = 192,001,536`` XLM-R input embedding matrix (the "192M" figure),
and re-enables gradients only on adapter parameters. Everything outside
``encoder.bert`` -- SuPar's scalar mix, the projection, and the ~8M parameter
span/label MLP head -- is left exactly as SuPar configured it.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

ADAPTER_NAME_MARKERS = ("lora_", "language_adapters", "adapter_")


def install_torch_load_compat(verbose: bool = True):
    r"""Let torch>=2.6 read SuPar 1.1.4 checkpoints.

    torch 2.6 flipped ``torch.load``'s ``weights_only`` default to ``True``, which
    refuses any pickle containing non-tensor globals. SuPar stores a ``Config``
    object and dill-pickles the whole ``Transform`` (fields and vocabularies) into
    its ``.pt`` files, so every load raises ``UnpicklingError``. Allow-listing the
    globals individually is impractical -- the transform pulls in a long tail of
    classes -- so we restore the previous default for this process.

    It also supplies ``map_location='cpu'`` when no GPU is available, because a
    checkpoint trained on a GPU stores cuda storage tags that torch will not
    deserialise onto a CPU-only process unaided.

    This is safe here precisely because the checkpoints are our own training
    output. Do not use it on a checkpoint from an untrusted source: full
    unpickling can execute arbitrary code.
    """
    import functools

    if getattr(torch.load, "_supar_compat", False):
        return torch.load

    original = torch.load

    @functools.wraps(original)
    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        if not torch.cuda.is_available():
            # Checkpoints trained on a GPU carry cuda storage tags, and torch
            # refuses to deserialise those without a map_location when no GPU is
            # available. SuPar calls torch.load with the path alone, so supply it.
            kwargs.setdefault("map_location", "cpu")
        return original(*args, **kwargs)

    load._supar_compat = True
    torch.load = load
    if verbose:
        print("[compat] torch.load: weights_only=False for SuPar checkpoints "
              f"(torch {torch.__version__})")
    return load


# ====================================================================== #
# Parameter bookkeeping
# ====================================================================== #
def is_adapter_param(name: str) -> bool:
    return any(marker in name for marker in ADAPTER_NAME_MARKERS)


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def describe_trainable(model: nn.Module, max_rows: int = 40) -> Dict[str, int]:
    r"""Print the trainable-parameter breakdown. Put this table in the report."""
    groups: Dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_adapter_param(name):
            key = "adapters"
        elif name.startswith("encoder.bert"):
            key = "backbone (LEAKED - should be empty)"
        elif name.startswith("encoder."):
            key = "scalar mix / projection"
        else:
            key = "parser head"
        groups[key] = groups.get(key, 0) + param.numel()

    total, trainable = count_parameters(model)
    print("\n" + "=" * 66)
    print("TRAINABLE PARAMETER BREAKDOWN")
    print("=" * 66)
    for key in sorted(groups, key=lambda k: -groups[k]):
        print(f"  {key:<38} {groups[key]:>12,}  ({100.0 * groups[key] / max(trainable, 1):5.1f}%)")
    print("-" * 66)
    print(f"  {'trainable':<38} {trainable:>12,}")
    print(f"  {'total':<38} {total:>12,}")
    print(f"  {'trainable fraction':<38} {100.0 * trainable / max(total, 1):>11.3f}%")
    print("=" * 66 + "\n")

    if any("LEAKED" in key for key in groups):
        print("WARNING: backbone parameters are still trainable. Freezing did not apply.\n")

    rows = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    if rows and len(rows) <= max_rows:
        for name, numel in rows:
            print(f"    {name:<58} {numel:>10,}")
        print()
    return groups


# ====================================================================== #
# Freezing
# ====================================================================== #
def freeze_backbone(bert: nn.Module, verbose: bool = True) -> Dict[str, int]:
    r"""Freeze the entire HF backbone, then unfreeze adapter parameters only."""
    frozen = 0
    for _name, param in bert.named_parameters():
        param.requires_grad = False
        frozen += param.numel()

    unfrozen = 0
    for name, param in bert.named_parameters():
        if is_adapter_param(name):
            param.requires_grad = True
            unfrozen += param.numel()

    embedding_params = 0
    embeddings = bert.get_input_embeddings() if hasattr(bert, "get_input_embeddings") else None
    if embeddings is not None:
        embedding_params = sum(p.numel() for p in embeddings.parameters())
        for param in embeddings.parameters():
            assert not param.requires_grad, "input embeddings must remain frozen"

    # SuPar sets this flag on the TransformerEmbedding wrapper; keep it honest.
    stats = {"frozen": frozen, "adapters_in_backbone": unfrozen, "embeddings": embedding_params}
    if verbose:
        print(f"[adapters] froze {frozen / 1e6:.2f}M backbone parameters "
              f"(input embeddings: {embedding_params:,} = {embedding_params / 1e6:.1f}M)")
        if unfrozen:
            print(f"[adapters] re-enabled {unfrozen:,} adapter parameters inside the backbone")
    return stats


# ====================================================================== #
# LoRA via peft
# ====================================================================== #
def add_lora_adapters(bert: nn.Module,
                      r: int = 16,
                      lora_alpha: int = 32,
                      lora_dropout: float = 0.1,
                      target_modules: Sequence[str] = ("query", "value"),
                      verbose: bool = True) -> nn.Module:
    r"""Inject LoRA into the frozen XLM-R backbone, in place.

    ``target_modules`` are matched as name suffixes by peft. For XLM-R,
    ``query``/``key``/``value`` are the attention projections. We avoid
    ``dense``, which is ambiguous (it names three different sublayers).
    """
    from peft import LoraConfig
    from peft.utils.peft_types import TaskType

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(target_modules),
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    try:
        from peft import inject_adapter_in_model

        bert = inject_adapter_in_model(config, bert)
        mode = "inject_adapter_in_model"
    except ImportError:  # very old peft
        from peft import get_peft_model

        bert = get_peft_model(bert, config)
        mode = "get_peft_model (PeftModel wrapper)"

    injected = sum(p.numel() for n, p in bert.named_parameters() if "lora_" in n)
    if verbose:
        n_layers = sum(1 for n, _ in bert.named_modules() if n.endswith("lora_A"))
        print(f"[adapters] LoRA via {mode}: r={r}, alpha={lora_alpha}, "
              f"dropout={lora_dropout}, targets={list(target_modules)}")
        print(f"[adapters] {n_layers} LoRA sites, {injected:,} LoRA parameters")
    return bert


# ====================================================================== #
# Pfeiffer bottleneck adapters (MAD-X), via forward hooks
# ====================================================================== #
class BottleneckAdapter(nn.Module):
    r"""Pfeiffer-style bottleneck: ``h + W_up(GELU(W_down(h)))``.

    ``W_up`` starts near zero so the module is an identity at initialisation.
    """

    def __init__(self, hidden_size: int, reduction_factor: int = 16, dropout: float = 0.0,
                 init_scale: float = 1e-3):
        super().__init__()
        bottleneck_size = max(8, hidden_size // reduction_factor)
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.normal_(self.down.weight, std=init_scale)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(self.dropout(self.act(self.down(hidden_states))))


def _get_encoder_layers(bert: nn.Module) -> List[nn.Module]:
    encoder = getattr(bert, "encoder", None)
    layers = getattr(encoder, "layer", None) if encoder is not None else None
    if layers is None:
        raise RuntimeError(
            f"Could not locate transformer layers on {type(bert).__name__}; "
            "expected `.encoder.layer` as in BERT/RoBERTa/XLM-R."
        )
    return list(layers)


def add_bottleneck_adapters(transformer_embedding: nn.Module,
                            reduction_factor: int = 16,
                            dropout: float = 0.0,
                            layers: Optional[Sequence[int]] = None,
                            verbose: bool = True) -> nn.ModuleList:
    r"""Attach one bottleneck adapter after each selected transformer layer.

    Args:
        transformer_embedding:
            ``parser.model.encoder`` (a ``supar.modules.pretrained.TransformerEmbedding``).
            The adapters are registered on it as ``language_adapters`` so that
            they land in the parser's ``state_dict`` and get moved by ``.to(device)``.
        layers:
            Indices of layers to adapt. ``None`` adapts all of them (MAD-X).
            Restricting to the top layers is a cheap ablation, e.g. ``range(8, 12)``.
    """
    bert = transformer_embedding.bert
    encoder_layers = _get_encoder_layers(bert)
    hidden_size = bert.config.hidden_size
    indices = list(range(len(encoder_layers))) if layers is None else list(layers)

    adapters = nn.ModuleList([
        BottleneckAdapter(hidden_size, reduction_factor, dropout) for _ in indices
    ])
    # Registering under this attribute name makes every parameter match
    # is_adapter_param() via the 'language_adapters' marker.
    transformer_embedding.language_adapters = adapters
    transformer_embedding.language_adapter_layers = indices

    handles = []
    for adapter, index in zip(adapters, indices):
        handles.append(encoder_layers[index].register_forward_hook(_make_hook(adapter)))
    transformer_embedding._adapter_hook_handles = handles

    n_params = sum(p.numel() for p in adapters.parameters())
    if verbose:
        bottleneck = max(8, hidden_size // reduction_factor)
        print(f"[adapters] {len(adapters)} Pfeiffer bottleneck adapters on layers {indices}: "
              f"{hidden_size}->{bottleneck}->{hidden_size}, {n_params:,} parameters "
              f"(up-projection zero-init => identity at step 0)")
    return adapters


def _make_hook(adapter: BottleneckAdapter):
    def hook(_module, _inputs, output):
        # HF encoder layers return a tuple whose first element is the hidden states.
        if isinstance(output, tuple):
            return (adapter(output[0]),) + tuple(output[1:])
        return adapter(output)

    return hook


# ====================================================================== #
# Top-level: patch SuPar's MODEL class
# ====================================================================== #
def patch_parser_with_adapters(parser_cls,
                               adapter_type: str = "lora",
                               lora_r: int = 16,
                               lora_alpha: int = 32,
                               lora_dropout: float = 0.1,
                               lora_targets: Sequence[str] = ("query", "value"),
                               reduction_factor: int = 16,
                               bottleneck_dropout: float = 0.0,
                               bottleneck_layers: Optional[Sequence[int]] = None,
                               gradient_checkpointing: bool = False,
                               verbose: bool = True):
    r"""Replace ``parser_cls.MODEL`` with a subclass that adapts + freezes the encoder.

    This must be called **before** ``parser_cls.build(...)`` *and* before any
    ``parser_cls.load(...)``: SuPar's ``Parser.load`` re-instantiates
    ``cls.MODEL(**args)`` and then calls ``load_state_dict(..., strict=False)``.
    If the adapters are absent at load time, the adapter tensors in the
    checkpoint are silently dropped (and for LoRA, whose injection renames
    ``query.weight`` to ``query.base_layer.weight``, the *backbone* weights would
    be silently dropped too). Use :func:`load_adapted_parser` for evaluation.
    """
    adapter_type = adapter_type.lower()
    if adapter_type not in ("lora", "bottleneck", "both"):
        raise ValueError(f"adapter_type must be lora|bottleneck|both, got {adapter_type!r}")

    OriginalModel = parser_cls.MODEL

    class AdaptedModel(OriginalModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            encoder = getattr(self, "encoder", None)
            bert = getattr(encoder, "bert", None)
            if bert is None:
                raise RuntimeError(
                    "Expected parser.model.encoder.bert (encoder='bert' in the SuPar config). "
                    f"Got encoder={type(encoder).__name__}."
                )

            if verbose:
                total, _ = count_parameters(bert)
                print(f"\n[adapters] adapting {type(bert).__name__} "
                      f"({total / 1e6:.1f}M parameters, {bert.config.num_hidden_layers} layers)")

            # 1. Freeze first, so any adapter created afterwards is trainable
            #    by construction and cannot be accidentally frozen.
            freeze_backbone(bert, verbose=verbose)

            # 2. Inject adapters.
            if adapter_type in ("lora", "both"):
                encoder.bert = add_lora_adapters(bert, r=lora_r, lora_alpha=lora_alpha,
                                                 lora_dropout=lora_dropout,
                                                 target_modules=lora_targets, verbose=verbose)
                bert = encoder.bert
                # peft freezes the base and marks lora_* trainable; assert it.
                freeze_backbone(bert, verbose=False)
                for name, param in bert.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

            if adapter_type in ("bottleneck", "both"):
                add_bottleneck_adapters(encoder, reduction_factor=reduction_factor,
                                        dropout=bottleneck_dropout,
                                        layers=bottleneck_layers, verbose=verbose)

            # 3. Keep SuPar's bookkeeping flag consistent with reality.
            encoder.requires_grad = False

            if gradient_checkpointing:
                enable_gradient_checkpointing(bert, verbose=verbose)

            if verbose:
                describe_trainable(self)

    AdaptedModel.__name__ = f"Adapted{OriginalModel.__name__}"
    AdaptedModel.__qualname__ = AdaptedModel.__name__
    parser_cls.MODEL = AdaptedModel
    if verbose:
        print(f"[adapters] {parser_cls.__name__}.MODEL -> {AdaptedModel.__name__} "
              f"(adapter_type={adapter_type})")
    return AdaptedModel


def enable_gradient_checkpointing(bert: nn.Module, verbose: bool = True):
    r"""Trade ~30% step time for a large activation-memory saving.

    Needed only if 12GB is tight at ``batch_size=2000`` tokens. ``use_reentrant=False``
    is required here: with frozen embeddings no *input* tensor requires grad, and
    the reentrant implementation would drop the graph entirely.
    """
    try:
        bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:  # transformers < 4.35
        bert.gradient_checkpointing_enable()
    if hasattr(bert, "enable_input_require_grads"):
        bert.enable_input_require_grads()
    if verbose:
        print("[adapters] gradient checkpointing enabled on the backbone")


# ====================================================================== #
# Optimizer: give the adapters their own learning rate
# ====================================================================== #
_ORIGINAL_ADAMW = None


def install_adapter_optimizer(model: nn.Module,
                              adapter_lr: float = 3e-4,
                              weight_decay: float = 0.0,
                              drop_frozen: bool = True,
                              verbose: bool = True):
    r"""Intercept the optimizer SuPar builds, and re-rate the adapter parameters.

    ``supar/parsers/parser.py`` builds one param group *per parameter*::

        from transformers import AdamW, get_linear_schedule_with_warmup
        self.optimizer = AdamW(
            [{'params': p, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
             for n, p in self.model.named_parameters()],
            args.lr)

    Two problems for us: (a) adapters live under ``encoder.bert.*``, so
    ``n.startswith('encoder')`` gives them ``args.lr`` (5e-5) -- far too low for
    randomly initialised LoRA/bottleneck weights, which typically want 1e-4..1e-3;
    (b) frozen parameters are handed to the optimizer too, bloating its state.

    Because the import happens *inside* ``train()``, rebinding
    ``transformers.AdamW`` before calling ``parser.train`` is enough -- no SuPar
    source edit. ``train_parser.py`` already rebinds this attribute, so we are
    extending an existing seam rather than inventing one.
    """
    global _ORIGINAL_ADAMW
    import transformers

    if _ORIGINAL_ADAMW is None:
        _ORIGINAL_ADAMW = getattr(transformers, "AdamW", None)

    name_by_id = {id(p): n for n, p in model.named_parameters()}
    report = {"adapters": 0, "other": 0, "dropped": 0}

    def AdamWFactory(params, lr=1e-3, *args, **kwargs):
        groups = []
        for group in params:
            if not isinstance(group, dict):
                group = {"params": group}
            tensors = group["params"]
            if isinstance(tensors, torch.Tensor):
                tensors = [tensors]
            for tensor in tensors:
                if drop_frozen and not tensor.requires_grad:
                    report["dropped"] += tensor.numel()
                    continue
                name = name_by_id.get(id(tensor), "")
                if is_adapter_param(name):
                    groups.append({"params": [tensor], "lr": adapter_lr,
                                   "weight_decay": weight_decay})
                    report["adapters"] += tensor.numel()
                else:
                    groups.append({"params": [tensor], "lr": group.get("lr", lr)})
                    report["other"] += tensor.numel()

        if not groups:
            raise RuntimeError("No trainable parameters reached the optimizer.")
        kwargs.pop("params", None)
        optimizer = torch.optim.AdamW(groups, lr=lr, **kwargs)
        if verbose:
            print(f"[adapters] optimizer: {report['adapters']:,} adapter params @ lr={adapter_lr:g}, "
                  f"{report['other']:,} head/mix params at SuPar's rates, "
                  f"{report['dropped']:,} frozen params excluded")
        return optimizer

    transformers.AdamW = AdamWFactory
    try:
        from transformers import optimization

        optimization.AdamW = AdamWFactory
    except ImportError:
        pass
    if verbose:
        print(f"[adapters] patched transformers.AdamW (adapter_lr={adapter_lr:g})")
    return AdamWFactory


# ====================================================================== #
# Loading an adapted checkpoint for evaluation
# ====================================================================== #
def load_adapted_parser(parser_cls, path: str, verbose: bool = True, **patch_kwargs):
    r"""Patch, then load, then *verify* -- because SuPar loads with strict=False.

    ``Parser.load`` calls ``model.load_state_dict(state['state_dict'], False)``.
    Silent key mismatches are the single most likely way to report a number that
    is actually a randomly initialised encoder, so this helper diffs the keys and
    raises if backbone weights failed to land.
    """
    patch_parser_with_adapters(parser_cls, verbose=verbose, **patch_kwargs)
    parser = parser_cls.load(path)

    state = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_keys = set(state["state_dict"].keys())
    model_keys = set(parser.model.state_dict().keys())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)

    substantive = [k for k in missing if "lora_" not in k and "language_adapters" not in k]
    if verbose:
        print(f"[adapters] checkpoint keys: {len(checkpoint_keys)}, model keys: {len(model_keys)}")
        if missing:
            print(f"[adapters] {len(missing)} keys not found in checkpoint (first 10): {missing[:10]}")
        if unexpected:
            print(f"[adapters] {len(unexpected)} unused checkpoint keys (first 10): {unexpected[:10]}")
    if substantive:
        raise RuntimeError(
            f"{len(substantive)} non-adapter parameters were NOT restored from {path}, "
            f"e.g. {substantive[:5]}. The adapter configuration used at load time almost "
            "certainly differs from the one used at train time."
        )
    return parser
