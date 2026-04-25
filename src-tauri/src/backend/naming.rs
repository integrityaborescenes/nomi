use once_cell::sync::Lazy;
use regex::Regex;

use super::ollama;

const MODEL: &str = "nomi-namer";
const MAX_ATTEMPTS: u8 = 5;

static PASCAL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Z][a-zA-Z]*$").unwrap());
static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Z][a-z]*").unwrap());
static EXTRACT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Z][A-Za-z]+").unwrap());

fn build_prompt(phrase: &str, word_count: u8, previous: &[String]) -> String {
    let mut s = format!("Phrase: {phrase}\nWords: {word_count}");
    if !previous.is_empty() {
        s.push_str(&format!("\nAvoid (do not repeat): {}", previous.join(", ")));
    }
    s
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

pub async fn generate_name(
    phrase: &str,
    word_count: u8,
    previous: &[String],
) -> Result<String, String> {
    if !(2..=5).contains(&word_count) {
        return Err("word_count must be between 2 and 5".into());
    }
    let phrase = phrase.trim();
    if phrase.is_empty() {
        return Err("phrase is empty".into());
    }

    let prompt = build_prompt(phrase, word_count, previous);
    let mut last_raw = String::new();

    for attempt in 0..MAX_ATTEMPTS {
        let temperature = (0.7 + (attempt as f32) * 0.1).min(1.1);
        let raw = ollama::generate(MODEL, &prompt, temperature).await?;
        last_raw = raw.clone();

        if let Some(candidate) = extract_candidate(&raw) {
            if count_words(&candidate) == word_count as usize
                && !previous.iter().any(|p| p == &candidate)
            {
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
