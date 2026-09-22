//! Laya's `DecisionModel`: a ModernBERT encoder (candle's implementation) followed by the
//! decision head from `laya/common.py`: a type embedding, `head_layers` pre-LN transformer
//! layers, a scorer read at each option's `[MASK]` marker, and the small `act_head`.
//!
//! Everything runs in F32. candle's ModernBERT builds its attention masks in F32, and the
//! Python runtime also uses F32 on CPU and MPS, so this matches the reference numerics.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, ops::softmax, Embedding, LayerNorm, Linear, VarBuilder,
};
use candle_transformers::models::modernbert::{self, ModernBert};

use crate::config::LayaConfig;
use crate::sequence::QType;

/// One `nn.TransformerEncoderLayer(norm_first=True)` with PyTorch's default ReLU activation.
struct HeadLayer {
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    n_heads: usize,
}

impl HeadLayer {
    fn load(vb: VarBuilder, d: usize, n_heads: usize) -> Result<Self> {
        let sa = vb.pp("self_attn");
        let in_w = sa.get((3 * d, d), "in_proj_weight")?;
        let in_b = sa.get(3 * d, "in_proj_bias")?;
        Ok(Self {
            in_proj: Linear::new(in_w, Some(in_b)),
            out_proj: linear(d, d, sa.pp("out_proj"))?,
            linear1: linear(d, 4 * d, vb.pp("linear1"))?,
            linear2: linear(4 * d, d, vb.pp("linear2"))?,
            norm1: layer_norm(d, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(d, 1e-5, vb.pp("norm2"))?,
            n_heads,
        })
    }

    /// `x + self_attn(norm1(x))`, then `x + linear2(relu(linear1(norm2(x))))`. Batch size is
    /// always 1 here, so there is no key padding to mask.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, d) = x.dims3()?;
        let dh = d / self.n_heads;
        let qkv = self.norm1.forward(x)?.apply(&self.in_proj)?;
        let split = |i: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(D::Minus1, i * d, d)?
                .reshape((b, l, self.n_heads, dh))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let scale = (dh as f64).powf(-0.5);
        let att = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let att = softmax(&att, D::Minus1)?;
        let ctx = att.matmul(&v)?.transpose(1, 2)?.reshape((b, l, d))?;
        let x = (x + ctx.apply(&self.out_proj)?)?;
        let ff = self
            .norm2
            .forward(&x)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        Ok((x + ff)?)
    }
}

/// Raw outputs for one question: one logit per option marker and the `act_head`'s
/// "act" probability.
#[derive(Debug, Clone)]
pub struct Scores {
    pub logits: Vec<f32>,
    pub act_probability: f32,
}

pub struct DecisionModel {
    encoder: ModernBert,
    head: Vec<HeadLayer>,
    type_emb: Embedding,
    scorer_norm: LayerNorm,
    scorer_l1: Linear,
    scorer_l2: Linear,
    act_l1: Linear,
    act_l2: Linear,
    device: Device,
}

impl DecisionModel {
    /// Load every weight from the checkpoint. The encoder tensors are stored under `encoder.`
    /// while candle expects `model.`, hence the rename.
    pub fn load(vb: VarBuilder, enc_cfg: &modernbert::Config, cfg: &LayaConfig) -> Result<Self> {
        let d = enc_cfg.hidden_size;
        let enc_vb = vb
            .clone()
            .rename_f(|name: &str| name.replacen("model.", "encoder.", 1));
        let encoder = ModernBert::load(enc_vb, enc_cfg)?;
        let n_heads = (d / 64).max(1);
        let mut head = Vec::with_capacity(cfg.head_layers);
        for i in 0..cfg.head_layers {
            head.push(HeadLayer::load(
                vb.pp(format!("head.layers.{i}")),
                d,
                n_heads,
            )?);
        }
        Ok(Self {
            encoder,
            head,
            type_emb: embedding(3, d, vb.pp("type_emb"))?,
            scorer_norm: layer_norm(d, 1e-5, vb.pp("scorer.0"))?,
            scorer_l1: linear(d, d, vb.pp("scorer.1"))?,
            scorer_l2: linear(d, 1, vb.pp("scorer.3"))?,
            act_l1: linear(d + 4, 256, vb.pp("act_head.0"))?,
            act_l2: linear(256, 2, vb.pp("act_head.2"))?,
            device: vb.device().clone(),
        })
    }

    /// Run one sequence. `markers` are the positions whose hidden state is scored.
    pub fn forward(&self, ids: &[u32], markers: &[usize], qtype: QType) -> Result<Scores> {
        if markers.is_empty() {
            return Err(anyhow!("a question needs at least one option marker"));
        }
        let l = ids.len();
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mask = Tensor::ones((1, l), DType::U32, &self.device)?;
        let mut h = self.encoder.forward(&input, &mask)?;
        let type_vec = self
            .type_emb
            .forward(&Tensor::new(&[qtype as u32], &self.device)?)?
            .unsqueeze(1)?;
        h = h.broadcast_add(&type_vec)?;
        for layer in &self.head {
            h = layer.forward(&h)?;
        }
        let marker_idx: Vec<u32> = markers.iter().map(|&m| m as u32).collect();
        let m = h.index_select(&Tensor::new(marker_idx.as_slice(), &self.device)?, 1)?;
        let logits = self
            .scorer_norm
            .forward(&m)?
            .apply(&self.scorer_l1)?
            .gelu_erf()?
            .apply(&self.scorer_l2)?
            .squeeze(D::Minus1)?; // [1, k]

        // act_head features, computed from the uncalibrated distribution as in the reference.
        let k = markers.len();
        let p = softmax(&logits, D::Minus1)?;
        let kf = (k.max(2)) as f64;
        let ent = (p.clone() * p.clamp(1e-9, f64::INFINITY)?.log()?)?
            .sum(D::Minus1)?
            .neg()?
            .affine(1.0 / kf.ln(), 0.0)?; // [1]
        let pv: Vec<f32> = p.i(0)?.to_vec1()?;
        let mut sorted = pv.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top1 = sorted[0];
        let top2 = if sorted.len() > 1 { sorted[1] } else { 0.0 };
        let ent_v: f32 = ent.to_vec1::<f32>()?[0];
        let feats = Tensor::new(
            &[[top1, top1 - top2, ent_v, (kf / 255.0) as f32]],
            &self.device,
        )?;
        let pooled = h.i((.., 0, ..))?; // [1, d]
        let act = Tensor::cat(&[pooled, feats], 1)?
            .apply(&self.act_l1)?
            .gelu_erf()?
            .apply(&self.act_l2)?;
        let act = softmax(&act, D::Minus1)?.i(0)?.to_vec1::<f32>()?;

        Ok(Scores {
            logits: logits.i(0)?.to_vec1()?,
            act_probability: act[0],
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
