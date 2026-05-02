use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{Device, IndexOp, Result as CResult, Tensor, D};
use candle_nn::{ops::softmax, Embedding, LayerNorm, Linear, Module};
use once_cell::sync::Lazy;
use rand::Rng;
use regex::Regex;
use sentencepiece::SentencePieceProcessor;
use serde::Deserialize;

const MODEL_BYTES: &[u8] = include_bytes!("../../resources/model.safetensors");
const CONFIG_BYTES: &[u8] = include_bytes!("../../resources/config.json");
const TOKENIZER_BYTES: &[u8] = include_bytes!("../../resources/tokenizer.model");

const MAX_ATTEMPTS: usize = 5;

#[derive(Debug, Deserialize)]
struct Config {
    vocab_size: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    max_seq: usize,
    bos_id: u32,
    eos_id: u32,
    sep_id: u32,
    wc_ids: HashMap<String, u32>,
}

struct Block {
    norm1: LayerNorm,
    attn_in_proj: Linear,
    attn_out_proj: Linear,
    norm2: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Block {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> CResult<Tensor> {
        let (b, t, d) = x.dims3()?;

        // attention block
        let h = self.norm1.forward(x)?;
        let qkv = self.attn_in_proj.forward(&h)?; // (B, T, 3d)
        let qkv = qkv.reshape((b, t, 3, self.n_heads, self.head_dim))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?; // (B, H, T, hd)
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        let scores = (scores / scale)?;
        let scores = scores.broadcast_add(mask)?; // mask: (T, T) с -inf над диагональю
        let attn = softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // (B, H, T, hd)
        let out = out.transpose(1, 2)?.reshape((b, t, d))?;
        let out = self.attn_out_proj.forward(&out)?;
        let x = (x + out)?;

        // feedforward block
        let h = self.norm2.forward(&x)?;
        let h = self.ff_in.forward(&h)?;
        let h = h.gelu()?;
        let h = self.ff_out.forward(&h)?;
        let x = (x + h)?;

        Ok(x)
    }
}

struct NanoDecoder {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    norm: LayerNorm,
    config: Config,
    device: Device,
    /// тензор tok_emb.weight для tied lm_head
    tok_emb_weight: Tensor,
}

impl NanoDecoder {
    fn load() -> CResult<Self> {
        let config: Config = serde_json::from_slice(CONFIG_BYTES)
            .map_err(|e| candle_core::Error::Msg(format!("config parse: {e}")))?;
        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load_buffer(MODEL_BYTES, &device)?;

        let get = |name: &str| -> CResult<Tensor> {
            tensors
                .get(name)
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("tensor not found: {name}")))
        };

        let tok_emb_w = get("tok_emb.weight")?;
        let tok_emb = Embedding::new(tok_emb_w.clone(), config.d_model);
        let pos_emb = Embedding::new(get("pos_emb.weight")?, config.d_model);

        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let p = format!("blocks.{i}");
            let norm1 = LayerNorm::new(
                get(&format!("{p}.norm1.weight"))?,
                get(&format!("{p}.norm1.bias"))?,
                1e-5,
            );
            let norm2 = LayerNorm::new(
                get(&format!("{p}.norm2.weight"))?,
                get(&format!("{p}.norm2.bias"))?,
                1e-5,
            );
            let attn_in_proj = Linear::new(
                get(&format!("{p}.attn.in_proj_weight"))?,
                Some(get(&format!("{p}.attn.in_proj_bias"))?),
            );
            let attn_out_proj = Linear::new(
                get(&format!("{p}.attn.out_proj.weight"))?,
                Some(get(&format!("{p}.attn.out_proj.bias"))?),
            );
            let ff_in = Linear::new(
                get(&format!("{p}.ff.0.weight"))?,
                Some(get(&format!("{p}.ff.0.bias"))?),
            );
            let ff_out = Linear::new(
                get(&format!("{p}.ff.3.weight"))?,
                Some(get(&format!("{p}.ff.3.bias"))?),
            );
            blocks.push(Block {
                norm1,
                attn_in_proj,
                attn_out_proj,
                norm2,
                ff_in,
                ff_out,
                n_heads: config.n_heads,
                head_dim: config.d_model / config.n_heads,
            });
        }

        let norm = LayerNorm::new(get("norm.weight")?, get("norm.bias")?, 1e-5);

        Ok(Self {
            tok_emb,
            pos_emb,
            blocks,
            norm,
            tok_emb_weight: tok_emb_w,
            config,
            device,
        })
    }

    /// Causal mask для длины T: -inf над диагональю, 0 на и под.
    fn causal_mask(&self, t: usize) -> CResult<Tensor> {
        let m: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_vec(m, (t, t), &self.device)
    }

    fn forward(&self, ids: &[u32]) -> CResult<Tensor> {
        let t = ids.len();
        let x = Tensor::from_vec(ids.to_vec(), (1, t), &self.device)?;
        let pos: Vec<u32> = (0..t as u32).collect();
        let pos = Tensor::from_vec(pos, (1, t), &self.device)?;

        let h = self.tok_emb.forward(&x)?;
        let p = self.pos_emb.forward(&pos)?;
        let mut h = (h + p)?;

        let mask = self.causal_mask(t)?;
        for blk in &self.blocks {
            h = blk.forward(&h, &mask)?;
        }
        let h = self.norm.forward(&h)?;

        // tied lm head: logits = h @ tok_emb.weight.T  →  (1, T, vocab)
        let logits = h.broadcast_matmul(&self.tok_emb_weight.t()?.contiguous()?)?;
        Ok(logits)
    }
}

static MODEL: Lazy<Mutex<NanoDecoder>> =
    Lazy::new(|| Mutex::new(NanoDecoder::load().expect("load model")));

static SP: Lazy<SentencePieceProcessor> = Lazy::new(|| {
    SentencePieceProcessor::from_serialized_proto(TOKENIZER_BYTES).expect("load tokenizer")
});

static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Z][a-z0-9]*").unwrap());
static PASCAL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^([A-Z][a-z0-9]*)+$").unwrap());

/// Возвращает ID токена ▁Word, если он есть в vocab.
fn word_token_id(word: &str) -> Option<u32> {
    let piece = format!("\u{2581}{word}");
    SP.piece_to_id(&piece).ok().flatten()
}

fn count_words(name: &str) -> usize {
    WORD_RE.find_iter(name).count()
}

fn unique_word_count(name: &str) -> usize {
    let words: std::collections::HashSet<_> = WORD_RE.find_iter(name).map(|m| m.as_str()).collect();
    words.len()
}

fn is_valid(name: &str, wc: usize) -> bool {
    if !PASCAL_RE.is_match(name) {
        return false;
    }
    let total = count_words(name);
    total == wc && unique_word_count(name) == wc
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

fn sample_with_temperature(logits: &[f32], indices: &[usize], temperature: f32) -> usize {
    // softmax над выбранными индексами (с температурой)
    let max_l = indices.iter().map(|&i| logits[i] / temperature).fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = indices
        .iter()
        .map(|&i| ((logits[i] / temperature) - max_l).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = probs.iter().map(|p| p / sum).collect();

    let r: f32 = rand::thread_rng().gen();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return indices[i];
        }
    }
    indices[indices.len() - 1]
}

fn generate_once(
    model: &NanoDecoder,
    phrase: &str,
    word_count: u32,
    temperature: f32,
    top_k: usize,
    previous: &[String],
) -> Result<String, String> {
    let cfg = &model.config;
    let wc_id = *cfg
        .wc_ids
        .get(&word_count.to_string())
        .ok_or_else(|| format!("wc {word_count} not in config"))?;

    let phrase_ids: Vec<u32> = SP
        .encode(phrase)
        .map_err(|e| format!("tokenize: {e}"))?
        .into_iter()
        .map(|p| p.id)
        .collect();

    let mut seq: Vec<u32> = vec![cfg.bos_id, wc_id];
    seq.extend(&phrase_ids);
    seq.push(cfg.sep_id);

    let mut out_tokens: Vec<u32> = Vec::new();
    let max_new = 24usize;

    for _ in 0..max_new {
        let logits = model
            .forward(&seq)
            .map_err(|e| format!("forward: {e}"))?;
        // logits: (1, T, vocab) — берём последнюю позицию
        let last = logits
            .i((0, seq.len() - 1, ..))
            .map_err(|e| format!("index: {e}"))?;
        let mut logits_vec: Vec<f32> = last.to_vec1().map_err(|e| format!("to_vec: {e}"))?;

        // декодим текущее имя
        let decoded = SP
            .decode_piece_ids(&out_tokens)
            .map_err(|e| format!("decode: {e}"))?
            .replace(' ', "");
        let n_words = count_words(&decoded);

        // блокируем EOS пока слов меньше wc
        if n_words < word_count as usize {
            logits_vec[cfg.eos_id as usize] = f32::NEG_INFINITY;
        }
        // блокируем уже использованные слова
        for w in WORD_RE.find_iter(&decoded) {
            if let Some(tid) = word_token_id(w.as_str()) {
                logits_vec[tid as usize] = f32::NEG_INFINITY;
            }
        }
        // на самом первом шаге блокируем первое слово ранее выданных результатов
        if out_tokens.is_empty() {
            for prev in previous {
                if let Some(first) = WORD_RE.find(prev) {
                    if let Some(tid) = word_token_id(first.as_str()) {
                        logits_vec[tid as usize] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        let topk = top_k_indices(&logits_vec, top_k);
        let next = sample_with_temperature(&logits_vec, &topk, temperature) as u32;

        if next == cfg.eos_id {
            break;
        }
        out_tokens.push(next);
        seq.push(next);

        // проверка после добавления
        let decoded = SP
            .decode_piece_ids(&out_tokens)
            .map_err(|e| format!("decode: {e}"))?
            .replace(' ', "");
        if count_words(&decoded) >= word_count as usize {
            break;
        }
        if seq.len() >= cfg.max_seq {
            break;
        }
    }

    let name = SP
        .decode_piece_ids(&out_tokens)
        .map_err(|e| format!("decode: {e}"))?
        .replace(' ', "");
    Ok(name)
}

pub fn generate_name(
    phrase: &str,
    word_count: u8,
    previous: &[String],
) -> Result<String, String> {
    if !(2..=4).contains(&word_count) {
        return Err("word_count must be between 2 and 4".into());
    }
    let phrase = phrase.trim();
    if phrase.is_empty() {
        return Err("phrase is empty".into());
    }

    let model = MODEL.lock().map_err(|e| format!("lock: {e}"))?;

    let mut last = String::new();
    for attempt in 0..MAX_ATTEMPTS {
        let temperature = 0.7 + 0.15 * attempt as f32;
        let name = generate_once(&model, phrase, word_count as u32, temperature, 15, previous)?;
        last = name.clone();
        if is_valid(&name, word_count as usize) && !previous.iter().any(|p| p == &name) {
            return Ok(name);
        }
    }
    if is_valid(&last, word_count as usize) {
        Ok(last)
    } else {
        Err(format!(
            "failed to produce valid {word_count}-word name after {MAX_ATTEMPTS} attempts; last: {last}"
        ))
    }
}
