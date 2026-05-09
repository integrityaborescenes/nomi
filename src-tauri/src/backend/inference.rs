use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::marian::{Config as MarianConfig, MTModel};
use once_cell::sync::Lazy;
use rand::Rng;
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;

const MODEL_BYTES: &[u8] = include_bytes!("../../resources/model.safetensors");
const CONFIG_BYTES: &[u8] = include_bytes!("../../resources/config.json");
// Marian имеет раздельные source (RU) и target (EN) SPM. Encode входа через source,
// decode выхода через target.
const TOKENIZER_SRC_BYTES: &[u8] = include_bytes!("../../resources/tokenizer-source.json");
const TOKENIZER_TGT_BYTES: &[u8] = include_bytes!("../../resources/tokenizer-target.json");

const MAX_ATTEMPTS: usize = 5;

#[derive(Debug, Deserialize)]
struct AppConfig {
    #[serde(flatten)]
    marian: MarianConfig,
    wc_ids: HashMap<String, u32>,
    max_decode_len: usize,
}

struct Engine {
    model: MTModel,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    config: AppConfig,
    device: Device,
}

impl Engine {
    fn load() -> Result<Self, String> {
        let config: AppConfig = serde_json::from_slice(CONFIG_BYTES)
            .map_err(|e| format!("config parse: {e}"))?;
        let device = Device::Cpu;
        // Веса хранятся в FP16 на диске (150 MB), но грузим как F32 в RAM —
        // candle::marian::decode() создаёт F32 attention mask, который конфликтует с F16.
        // RAM при работе ~300 MB, exe-файл всё равно ~150 MB.
        let vb = VarBuilder::from_slice_safetensors(MODEL_BYTES, DType::F32, &device)
            .map_err(|e| format!("varbuilder: {e}"))?;
        let model = MTModel::new(&config.marian, vb)
            .map_err(|e| format!("model new: {e}"))?;
        let src_tokenizer = Tokenizer::from_bytes(TOKENIZER_SRC_BYTES)
            .map_err(|e| format!("src tokenizer: {e}"))?;
        let tgt_tokenizer = Tokenizer::from_bytes(TOKENIZER_TGT_BYTES)
            .map_err(|e| format!("tgt tokenizer: {e}"))?;
        Ok(Self {
            model,
            src_tokenizer,
            tgt_tokenizer,
            config,
            device,
        })
    }

    /// Один прогон генерации: encode source → decode loop с применёнными constraints.
    fn generate_once(
        &mut self,
        phrase: &str,
        word_count: u32,
        temperature: f32,
        previous: &[String],
    ) -> Result<String, String> {
        let cfg = &self.config;
        let wc_id = *cfg
            .wc_ids
            .get(&word_count.to_string())
            .ok_or_else(|| format!("wc {word_count} not in config"))?;

        // Source: «<wcN> phrase </s>» — encode + добавить wc_id в начало + EOS в конец.
        let encoded = self
            .src_tokenizer
            .encode(phrase.to_string(), false) // false — special-токены добавим сами
            .map_err(|e| format!("tokenize: {e}"))?;
        let mut src_ids: Vec<u32> = vec![wc_id];
        src_ids.extend_from_slice(encoded.get_ids());
        src_ids.push(cfg.marian.eos_token_id);

        let src_tensor = Tensor::new(src_ids.as_slice(), &self.device)
            .map_err(|e| format!("src tensor: {e}"))?
            .unsqueeze(0)
            .map_err(|e| format!("unsqueeze: {e}"))?;

        // Сбрасываем kv-cache между генерациями.
        self.model.reset_kv_cache();

        // Encoder forward (один раз).
        let encoder_out = self
            .model
            .encoder()
            .forward(&src_tensor, 0)
            .map_err(|e| format!("encoder: {e}"))?;

        // Подготовим constraints — IDs spec-токенов и uppercase-word-piece IDs.
        let eos_id = cfg.marian.eos_token_id;
        let pad_id = cfg.marian.pad_token_id;
        let start_id = cfg.marian.decoder_start_token_id;

        // Слова — первое слово предыдущих результатов (для разнообразия).
        let mut blocked_first_words: Vec<u32> = Vec::new();
        for prev in previous {
            if let Some(first) = WORD_RE.find(prev) {
                if let Some(tid) = self.word_token_id(first.as_str()) {
                    blocked_first_words.push(tid);
                }
            }
        }

        // RNG seed для LogitsProcessor.
        let seed: u64 = rand::thread_rng().gen();
        let mut sampler = LogitsProcessor::from_sampling(
            seed,
            Sampling::TopK {
                k: 15,
                temperature: temperature as f64,
            },
        );

        // Авторегрессивный декод с kv-cache.
        // На первом шаге подаём [start_id] с past_kv_len=0.
        // На последующих — только новый токен с past_kv_len=кол-во ранее обработанных.
        let mut output_ids: Vec<u32> = Vec::new();
        let mut current_input: Vec<u32> = vec![start_id];
        let mut past_kv_len: usize = 0;

        for step in 0..cfg.max_decode_len {
            let dec_input = Tensor::new(current_input.as_slice(), &self.device)
                .map_err(|e| format!("dec tensor: {e}"))?
                .unsqueeze(0)
                .map_err(|e| format!("unsqueeze: {e}"))?;

            let logits = self
                .model
                .decode(&dec_input, &encoder_out, past_kv_len)
                .map_err(|e| format!("decode: {e}"))?;

            // logits: (1, current_input.len(), vocab) — берём последнюю позицию.
            let last_logits = logits
                .i((0, current_input.len() - 1, ..))
                .map_err(|e| format!("index: {e}"))?
                .to_dtype(DType::F32)
                .map_err(|e| format!("to_dtype: {e}"))?;

            // Текущее накопленное имя для constraints (decode через target).
            let accumulated_text = if output_ids.is_empty() {
                String::new()
            } else {
                self.tgt_tokenizer
                    .decode(&output_ids, true)
                    .map_err(|e| format!("decode: {e}"))?
            };
            let n_words = count_words(&accumulated_text);
            let used_words: Vec<u32> = WORD_RE
                .find_iter(&accumulated_text)
                .filter_map(|m| self.word_token_id(m.as_str()))
                .collect();
            let block_first_step = step == 0;

            let next = sampler
                .sample_f(&last_logits, |prs| {
                    // 1. Блокируем pad всегда.
                    if (pad_id as usize) < prs.len() {
                        prs[pad_id as usize] = 0.0;
                    }
                    // 2. Блокируем EOS пока не сгенерили нужное число слов.
                    if n_words < word_count as usize && (eos_id as usize) < prs.len() {
                        prs[eos_id as usize] = 0.0;
                    }
                    // 3. Блокируем уже использованные слова.
                    for &tid in &used_words {
                        if (tid as usize) < prs.len() {
                            prs[tid as usize] = 0.0;
                        }
                    }
                    // 4. На первом шаге блокируем первое слово предыдущих результатов.
                    if block_first_step {
                        for &tid in &blocked_first_words {
                            if (tid as usize) < prs.len() {
                                prs[tid as usize] = 0.0;
                            }
                        }
                    }
                })
                .map_err(|e| format!("sample: {e}"))?;

            if next == eos_id {
                break;
            }
            output_ids.push(next);

            // Двигаем кэш: следующий шаг подаст только [next], past_kv_len растёт.
            past_kv_len += current_input.len();
            current_input = vec![next];

            // Не прерываемся на достижении word_count — иначе можем отрезать
            // суффикс слова (Footer → "F" + "oot" + "er", break после "F" даст "SiteF").
            // Полагаемся на natural EOS: как только n_words ≥ word_count, EOS
            // разблокируется и модель его эмитит сама когда закончит.
        }

        // Финальный декод и приведение к PascalCase (decode через target).
        let raw = self
            .tgt_tokenizer
            .decode(&output_ids, true)
            .map_err(|e| format!("final decode: {e}"))?;
        Ok(to_pascal_case(&raw))
    }

    /// Получить ID токена представляющего отдельное слово (uppercase) — для constraint-логики.
    /// В opus-mt SPM-токены либо начинаются с ▁ (новое слово), либо являются продолжением.
    /// Для блокировки повторов нам нужен ID токена «▁Word».
    fn word_token_id(&self, word: &str) -> Option<u32> {
        // Слова — английские, ищем через target tokenizer.
        let with_marker = format!("\u{2581}{word}");
        self.tgt_tokenizer
            .token_to_id(&with_marker)
            .or_else(|| self.tgt_tokenizer.token_to_id(word))
    }
}

static ENGINE: Lazy<Mutex<Engine>> =
    Lazy::new(|| Mutex::new(Engine::load().expect("load marian engine")));

static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z][a-z0-9]*").unwrap());
static PASCAL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^([A-Z][a-z0-9]*)+$").unwrap());

fn count_words(text: &str) -> usize {
    WORD_RE.find_iter(text).count()
}

fn unique_word_count(text: &str) -> usize {
    let words: std::collections::HashSet<_> =
        WORD_RE.find_iter(text).map(|m| m.as_str()).collect();
    words.len()
}

fn is_valid(name: &str, wc: usize) -> bool {
    if !PASCAL_RE.is_match(name) {
        return false;
    }
    let total = count_words(name);
    total == wc && unique_word_count(name) == wc
}

/// «product card» / «Product card» / «product Card» → «ProductCard»
fn to_pascal_case(text: &str) -> String {
    text.split_whitespace()
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
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

    let mut engine = ENGINE.lock().map_err(|e| format!("lock: {e}"))?;

    let mut last = String::new();
    for attempt in 0..MAX_ATTEMPTS {
        let temperature = 0.7 + 0.15 * attempt as f32;
        let name = engine.generate_once(phrase, word_count as u32, temperature, previous)?;
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
