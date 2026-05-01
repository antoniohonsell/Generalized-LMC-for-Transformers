from itertools import chain
import argparse
import json
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    DataCollatorForLanguageModeling,
)

import wandb


# ── Muon optimizer ────────────────────────────────────────────────────────────

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Orthogonalize G via quintic Newton-Schulz iteration."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon — MomentUm Orthogonalized by Newton-schulz.

    Applies Nesterov SGD momentum then replaces each 2-D update with its
    nearest orthogonal matrix (Newton-Schulz), scaled to preserve the RMS
    gradient norm.  Only pass 2-D weight tensors (no embeddings, biases, or
    layer-norm params — use AdamW for those).

    Reference: https://github.com/KellerJordan/modded-nanogpt
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]

                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)

                buf = state["buf"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = max(1.0, update.size(0) / update.size(1)) ** 0.5
                p.add_(update.to(p.dtype), alpha=-lr * scale)


# ── Trainer subclass ──────────────────────────────────────────────────────────

class MuonTrainer(Trainer):
    """
    Trainer that uses Muon for 2-D weight matrices and AdamW for everything else.

    Rather than wrapping both optimizers into one (which fights with how HF
    Trainer / accelerate internally manages optimizers), we give Trainer a plain
    AdamW for its native optimizer slot and step Muon manually right after the
    backward pass inside training_step — at which point gradients are ready and
    we still own the call stack.
    """

    def __init__(self, *args, muon_lr: float = 0.02, muon_momentum: float = 0.95, **kwargs):
        self._muon_lr       = muon_lr
        self._muon_momentum = muon_momentum
        self._muon_opt      = None  # created in create_optimizer
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        model = self.model

        seen_ids: set = set()
        muon_params, adamw_params = [], []

        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in seen_ids:
                continue
            seen_ids.add(id(param))

            if (param.ndim == 2
                    and "wte"    not in name
                    and "wpe"    not in name
                    and "lm_head" not in name):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        # Muon is stepped manually in training_step; Trainer owns AdamW.
        self._muon_opt = Muon(muon_params, lr=self._muon_lr, momentum=self._muon_momentum)
        self.optimizer = torch.optim.AdamW(
            adamw_params,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def training_step(self, model, inputs, *args, **kwargs):
        # super() handles forward + backward; gradients are ready when it returns.
        # Forward any extra positional/keyword args (e.g. num_items_in_batch in
        # newer Transformers versions) so we stay forward-compatible.
        loss = super().training_step(model, inputs, *args, **kwargs)
        # Step Muon immediately after backward.
        # AdamW is stepped by Trainer's outer loop via self.optimizer.
        self._muon_opt.step()
        self._muon_opt.zero_grad()
        return loss


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_dir",    type=str, required=True)
    p.add_argument("--tokenizer_dir", type=str, default="./gpt2_tokenizer")
    p.add_argument("--seed",          type=int, default=1)
    p.add_argument("--block_size",    type=int, default=256)
    p.add_argument("--n_layer",       type=int, default=6)
    p.add_argument("--n_embd",        type=int, default=256)
    p.add_argument("--n_inner",       type=int, default=1024)
    p.add_argument("--n_head",        type=int, default=4)
    p.add_argument("--tie_word_embeddings", action="store_true")
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=3e-4,  help="AdamW learning rate.")
    p.add_argument("--muon_lr",       type=float, default=0.02,  help="Muon learning rate.")
    p.add_argument("--muon_momentum", type=float, default=0.95,  help="Muon momentum.")
    p.add_argument("--warmup_ratio",  type=float, default=0.05)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--eval_steps",    type=int,   default=50)
    p.add_argument("--logging_steps", type=int,   default=25)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--early_stop",    action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--fp16",          action="store_true")
    p.add_argument("--output_dir",    type=str, default=None)
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_project", type=str, default="gpt2-merging-demo")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--wandb_group",   type=str, default=None)
    p.add_argument("--wandb_tags",    type=str, default="merge")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.output_dir or f"gpt2_muon_tinyshakespeare_seed{args.seed}_nembd{args.n_embd}"
    os.makedirs(out_dir, exist_ok=True)

    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="train",
            config=vars(args),
            tags=[t for t in args.wandb_tags.split(",") if t],
            name=f"gpt2-muon-seed{args.seed}",
        )

    ds = load_from_disk(args.splits_dir)
    raw_train, raw_val, raw_test = ds["train"], ds["validation"], ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token
    block_size = args.block_size

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=True)

    def group_texts(examples):
        ids   = list(chain(*examples["input_ids"]))
        masks = list(chain(*examples["attention_mask"]))
        chunk_len = (len(ids) // block_size) * block_size
        ids, masks = ids[:chunk_len], masks[:chunk_len]
        return {
            "input_ids":      [ids[i:i+block_size]   for i in range(0, chunk_len, block_size)],
            "attention_mask": [masks[i:i+block_size]  for i in range(0, chunk_len, block_size)],
        }

    def prep(ds_split):
        t = ds_split.map(tokenize_fn, batched=True, remove_columns=["text"])
        c = t.map(group_texts, batched=True)
        c.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return c

    chunked_train = prep(raw_train)
    chunked_val   = prep(raw_val)
    chunked_test  = prep(raw_test)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    configuration = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner,
        tie_word_embeddings=args.tie_word_embeddings,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(configuration)

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,          # used by AdamW group
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        report_to=("wandb" if args.wandb else "none"),
        run_name=f"gpt2-muon-seed{args.seed}-n_embd{args.n_embd}",
    )

    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

    trainer = MuonTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=chunked_train,
        eval_dataset=chunked_val,
        data_collator=data_collator,
        callbacks=callbacks,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
    )

    trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt is not None:
        with open(os.path.join(out_dir, "BEST_CHECKPOINT.txt"), "w") as f:
            f.write(best_ckpt + "\n")

    test_metrics = trainer.evaluate(eval_dataset=chunked_test, metric_key_prefix="test")
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("TEST metrics:", test_metrics)

    if run is not None:
        art = wandb.Artifact(
            name=f"gpt2-muon-model-seed{args.seed}",
            type="model",
            metadata={
                "seed":          args.seed,
                "block_size":    args.block_size,
                "n_layer":       args.n_layer,
                "n_embd":        args.n_embd,
                "n_head":        args.n_head,
                "muon_lr":       args.muon_lr,
                "muon_momentum": args.muon_momentum,
                "splits_dir":    args.splits_dir,
                "best_checkpoint": best_ckpt,
            },
        )
        art.add_dir(out_dir)
        aliases = ["latest", f"seed-{args.seed}"]
        if best_ckpt:
            aliases.append("best")
        run.log_artifact(art, aliases=aliases)
        run.finish()


if __name__ == "__main__":
    main()
