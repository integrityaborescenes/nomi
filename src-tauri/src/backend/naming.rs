use once_cell::sync::Lazy;
use regex::Regex;

use super::ollama;

const MODEL: &str = "qwen2.5:3b-instruct";
const MAX_ATTEMPTS: u8 = 3;

static PASCAL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Z][a-zA-Z]*$").unwrap());
static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Z][a-z]*").unwrap());
static EXTRACT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Z][A-Za-z]+").unwrap());

fn build_prompt(phrase: &str, word_count: u8) -> String {
    let example = match word_count {
        2 => "ApplyFilters",
        3 => "ApplyPhotoFilters",
        4 => "ApplyPhotoStripFilters",
        _ => "ApplyPhotoFilters",
    };

    format!(
        "You translate phrases (often Russian) into a single PascalCase identifier suitable for a React component name.\n\
        \n\
        STRICT RULES:\n\
        - Output ONLY the identifier. No quotes, no markdown, no explanation, no extra words.\n\
        - The identifier MUST contain EXACTLY {word_count} English words concatenated in PascalCase.\n\
        - Each word: first letter uppercase, rest lowercase. ASCII letters only.\n\
        - Do not append suffixes like \"Component\", \"Page\", \"Button\" unless the phrase mentions them.\n\
        - Choose words that best capture the meaning of the phrase.\n\
        \n\
        Example for {word_count} words: {example}\n\
        \n\
        Phrase: {phrase}\n\
        \n\
        Identifier:"
    )
}

fn extract_candidate(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let first_line = trimmed.lines().next()?.trim();
    let cleaned: String = first_line
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string();

    if PASCAL_RE.is_match(&cleaned) {
        return Some(cleaned);
    }
    EXTRACT_RE
        .find_iter(first_line)
        .map(|m| m.as_str().to_string())
        .max_by_key(|s| s.len())
}

fn count_words(identifier: &str) -> usize {
    WORD_RE.find_iter(identifier).count()
}

pub async fn generate_name(phrase: &str, word_count: u8) -> Result<String, String> {
    if !(2..=4).contains(&word_count) {
        return Err("word_count must be 2, 3, or 4".into());
    }
    let phrase = phrase.trim();
    if phrase.is_empty() {
        return Err("phrase is empty".into());
    }

    let prompt = build_prompt(phrase, word_count);
    let mut last_raw = String::new();

    for attempt in 0..MAX_ATTEMPTS {
        let temperature = 0.6 + (attempt as f32) * 0.2;
        let raw = ollama::generate(MODEL, &prompt, temperature).await?;
        last_raw = raw.clone();

        if let Some(candidate) = extract_candidate(&raw) {
            if count_words(&candidate) == word_count as usize {
                return Ok(candidate);
            }
        }
    }

    Err(format!(
        "model failed to produce a valid {word_count}-word PascalCase name after {MAX_ATTEMPTS} attempts. last response: {last_raw}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_clean_identifier() {
        assert_eq!(extract_candidate("ApplyPhotoFilters"), Some("ApplyPhotoFilters".into()));
    }

    #[test]
    fn strips_quotes_and_punctuation() {
        assert_eq!(extract_candidate("\"ApplyPhotoFilters\"."), Some("ApplyPhotoFilters".into()));
    }

    #[test]
    fn extracts_from_noisy_response() {
        let raw = "Here is the identifier: ApplyPhotoFilters - hope this helps!";
        assert_eq!(extract_candidate(raw), Some("ApplyPhotoFilters".into()));
    }

    #[test]
    fn counts_words_correctly() {
        assert_eq!(count_words("ApplyPhotoFilters"), 3);
        assert_eq!(count_words("Apply"), 1);
        assert_eq!(count_words("ApplyPhotoStripFilters"), 4);
    }
}
