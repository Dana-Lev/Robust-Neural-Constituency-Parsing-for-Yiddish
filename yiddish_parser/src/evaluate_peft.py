# -*- coding: utf-8 -*-
r"""
Evaluate an adapter-trained checkpoint and print Labelled F1.

Needed as a separate entrypoint because ``supar.parsers.parser.Parser.load``
does ``model.load_state_dict(state['state_dict'], False)`` -- strict=False. If
``CRFConstituencyParser.MODEL`` is not patched with the same adapter
configuration that was used for training, the adapter tensors (and, for LoRA,
the renamed ``*.base_layer.weight`` backbone tensors) are dropped in silence and
you evaluate a partially random encoder that still reports a plausible-looking
number. ``language_adapters.load_adapted_parser`` diffs the key sets and raises
instead.

    python src/evaluate_peft.py --path ./output/parser_both_lora/yiddish_parser.pt \
                                --adapter-type lora --data data/processed/supar_ready/test.txt
"""

import argparse
import os
import sys


def _early_device_choice():
    """Read --device before torch initialises CUDA.

    SuPar's ``Parser.load`` sets ``args.device = 'cuda' if
    torch.cuda.is_available() else 'cpu'`` *after* consuming its keyword
    arguments, so the device cannot be requested through the API. The only
    reliable lever is what torch reports, and for an explicit CPU request the
    cleanest one is to hide the GPUs before torch ever looks.
    """
    argv = sys.argv
    choice = "auto"
    if "--device" in argv:
        index = argv.index("--device")
        if index + 1 < len(argv):
            choice = argv[index + 1]
    if choice == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return choice


_DEVICE_CHOICE = _early_device_choice()

import torch  # noqa: E402 - must follow the CUDA_VISIBLE_DEVICES decision
import transformers  # noqa: E402

transformers.AdamW = torch.optim.AdamW

from supar import CRFConstituencyParser  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import language_adapters as la  # noqa: E402


def resolve_device():
    """Pick a device that can actually run kernels, and tell SuPar about it.

    Evaluation is cheap enough to do on CPU, so unlike training we fall back
    rather than refusing: a wheel with no kernels for the visible GPU's compute
    capability would otherwise crash inside the first batch.
    """
    if _DEVICE_CHOICE == "cpu":
        print("[device] cpu (requested)")
        return "cpu"
    if not torch.cuda.is_available():
        print("[device] cpu (no GPU visible)")
        return "cpu"

    name = torch.cuda.get_device_name(0)
    try:
        probe = torch.zeros(64, device="cuda")
        probe.add_(1.0)
        (probe @ probe).item()
        print(f"[device] cuda ({name}) -- kernel launch OK")
        return "cuda"
    except Exception as exc:  # noqa: BLE001 - fall back rather than crash
        capability = "sm_%d%d" % torch.cuda.get_device_capability(0)
        arches = " ".join(getattr(torch.cuda, "get_arch_list", lambda: [])())
        print(f"[device] {name} is {capability}, but this torch builds kernels "
              f"for {arches or 'unknown'}.")
        print(f"[device] {type(exc).__name__}: {str(exc).splitlines()[0]}")
        print("[device] falling back to cpu. Evaluation is cheap; for a GPU run "
              "use an allocation on a supported card, e.g.")
        print("[device]   srun -p studentkillable --constraint=geforce_rtx_2080 "
              "--gpus=1 --pty python src/evaluate_peft.py ...")
        # SuPar consults this inside Parser.load, so it must answer honestly.
        torch.cuda.is_available = lambda: False
        return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Path to yiddish_parser.pt")
    p.add_argument("--data", default="data/processed/supar_ready/test.txt")
    p.add_argument("--adapter-type", choices=["none", "lora", "bottleneck", "both"], default="lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-targets", nargs="+", default=["query", "value"])
    p.add_argument("--reduction-factor", type=int, default=16)
    p.add_argument("--adapter-layers", nargs="+", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--buckets", type=int, default=8)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="'auto' uses the GPU only if it can actually run kernels, "
                        "otherwise falls back to CPU. 'cpu' hides the GPUs entirely.")
    args = p.parse_args()

    # torch>=2.6 refuses SuPar's checkpoints by default; see the docstring.
    la.install_torch_load_compat()
    resolve_device()

    if args.adapter_type == "none":
        parser = CRFConstituencyParser.load(args.path)
    else:
        parser = la.load_adapted_parser(
            CRFConstituencyParser, args.path,
            adapter_type=args.adapter_type,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            lora_targets=args.lora_targets,
            reduction_factor=args.reduction_factor,
            bottleneck_layers=args.adapter_layers)

    # Segmentation is deterministic at evaluation time: the stochastic collator
    # is only installed on loaders built with shuffle=True.
    loss, metric = parser.evaluate(args.data, buckets=args.buckets, batch_size=args.batch_size)

    print("\n" + "=" * 66)
    print(f"EVALUATION: {os.path.basename(os.path.dirname(args.path))}  on  {args.data}")
    print("=" * 66)
    print(f"  loss            {loss:.4f}")
    print(f"  metric          {metric}")
    for attribute, label in (("lp", "Labelled Precision"), ("lr", "Labelled Recall"),
                             ("lf", "Labelled F1"), ("up", "Unlabelled Precision"),
                             ("ur", "Unlabelled Recall"), ("uf", "Unlabelled F1"),
                             ("lcm", "Labelled Complete Match"), ("ucm", "Unlabelled Complete Match")):
        value = getattr(metric, attribute, None)
        if value is not None:
            print(f"  {label:<26} {100 * float(value):6.2f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
